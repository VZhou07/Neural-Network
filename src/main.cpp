#include <iostream>
#include <raylib.h>
#include "neural_net.hpp"
#include <thread>
#include <cmath>
#include <Eigen/Dense>

std::atomic<bool> done_training=false;

void training_thread(NeuralNet& net){
    bool success=net.load("data/best_model.bin");
    if (!success){
        net.reinitialize_weights();
        float learning_rate=0.005;
        net.train(learning_rate);
    }
    float accuracy=net.test_accuracy();
    std::cout << "Accuracy: " << accuracy << std::endl;
    done_training=true;
}

int main() {
    NeuralNet net({784, 256, 64, 10}, 32,true,true);
    const int width=1280;
    const int height=720;
    InitWindow(width, height, "Number Classifier");
    std::thread t(training_thread, std::ref(net));
    float brush_size=10;
    SetTargetFPS(120);
    Rectangle draw_rectangle={10, 30, 280, 280};
    RenderTexture2D target=LoadRenderTexture(280, 280);
    BeginTextureMode(target);
    ClearBackground(BLACK);
    EndTextureMode();
    Texture2D preview_texture={0,0,0,0};
    bool has_preview=false;
    int prediction=-1;
    bool display_not_done=false;
    Vector2 prev={0,0};
    bool has_prev=false;
    unsigned char* input_data=new unsigned char[784];
    while (!WindowShouldClose()){
        Eigen::MatrixXd processed_image=Eigen::MatrixXd::Zero(784,1);
        Vector2 mouse_position=GetMousePosition();
        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)){
            has_prev=false;
        }
        if (CheckCollisionPointRec(mouse_position, draw_rectangle)){
            if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)){
                Vector2 local_mouse_position={mouse_position.x-draw_rectangle.x, mouse_position.y-draw_rectangle.y};
                BeginTextureMode(target);
                if (has_prev){
                    DrawLineEx(prev, local_mouse_position, brush_size*2, WHITE);
                }
                prev=local_mouse_position;
                has_prev=true;
                DrawCircle(local_mouse_position.x, local_mouse_position.y, brush_size, WHITE);
                EndTextureMode();
            }
        }
        if (IsKeyPressed(KEY_C)){
            BeginTextureMode(target);
            ClearBackground(BLACK);
            EndTextureMode();
        }
        if (IsKeyPressed(KEY_Q)){
            break;
        }

        if (IsKeyPressed(KEY_ENTER) && done_training){
            Image image=LoadImageFromTexture(target.texture);
            ImageFlipVertical(&image);
            int y_max=-1;
            int x_max=-1;
            int y_min=281;
            int x_min=281;
            const unsigned char* texture_data=static_cast<const unsigned char*>(image.data);
            for(int i=0;i<image.height;i++){
                for(int j=0;j<image.width;j++){
                    int index=(i*280+j)*4;
                    unsigned char r=texture_data[index];
                    if (r>=250){
                        if (i<y_min){
                            y_min=i;
                        }
                        if (i>y_max){
                            y_max=i;
                        }
                        if (j<x_min){
                            x_min=j;
                        }
                        if (j>x_max){
                            x_max=j;
                        }
                    }
                }
            }
            float width=static_cast<float>(x_max-x_min)+1.0;
            float height=static_cast<float>(y_max-y_min)+1.0;
            if (width<0||height<0){
                for(int i=0;i<784;i++){
                    input_data[i]=0;
                    processed_image(i,0)=0;
                }
            }
            else{

                ImageCrop(&image, Rectangle{static_cast<float>(x_min), static_cast<float>(y_min), width, height});            
                float width_ratio=float(image.width)/20;
                float height_ratio=float(image.height)/20;
                
                if (width_ratio>height_ratio){
                    ImageResize(&image, 20, std::round(image.height/width_ratio));
                }
                else{
                    ImageResize(&image, image.width/height_ratio, 20);
                }
                
                const unsigned char* image_data=static_cast<const unsigned char*>(image.data); //gives rgba
                float x_moment=0;
                float y_moment=0;
                float total_mass=0;
                for(int i=0;i<image.height;i++){
                    for(int j=0;j<image.width;j++){
                        int index=(i*image.width+j)*4;
                        unsigned char r=image_data[index];
                        x_moment+=r/255.0*j;
                        y_moment+=r/255.0*i;
                        total_mass+=r/255.0;
                    }
                }
                float center_x=total_mass>0?x_moment/total_mass:0;
                float center_y=total_mass>0?y_moment/total_mass:0;

                int shift_x=std::round(13.5-center_x); //13.5 is the center of the image
                int shift_y=std::round(13.5-center_y);
                int count=0;
                for (int i=0;i<image.height;i++){
                    for(int j=0;j<image.width;j++){
                        int target_x=j+shift_x;
                        int target_y=i+shift_y;
                        if (target_x>=0 && target_x<28 && target_y>=0 && target_y<28){
                            processed_image(target_y*28+target_x,0)=image_data[(i*image.width+j)*4]/255.0;
                        }
                    }
                }
                for(int i=0;i<784;i++){
                    input_data[i]=static_cast<unsigned char>(processed_image(i,0)*255.0);
                }
            }

            Image display = {
                .data=input_data,
                .width=28,
                .height=28,
                .mipmaps=1,
                .format=PIXELFORMAT_UNCOMPRESSED_GRAYSCALE,
            };
            std::cout << "Processed Image: " << processed_image << std::endl;
            prediction=net.test(processed_image);
            std::cout << "Prediction: " << prediction << std::endl;
            if (has_preview){
                UnloadTexture(preview_texture);
            }
            preview_texture=LoadTextureFromImage(display);
            has_preview=true;
            UnloadImage(image);
            }
            
        BeginDrawing();
        ClearBackground(GRAY);
        if (!done_training){
            DrawText("Training not done, please wait", 200, 10, 20, BLACK);
        }
        else{
            DrawText("Click Enter to Submit Drawing", 200, 10, 20, BLACK);
        }
        DrawRectangleLinesEx(draw_rectangle, 2, BLACK);
        DrawTextureRec(target.texture, {0, 0, 280, -280}, {10, 30}, WHITE);
        if (has_preview){
            DrawTexture(preview_texture, 700, 30, WHITE);
            if (prediction!=-1){
                std::string prediction_string="prediction: "+std::to_string(prediction);
                DrawText(prediction_string.c_str(), 700, 30+preview_texture.height+10, 20, BLACK);
            }
        }
        EndDrawing();
    }
    t.join();
    delete[] input_data;
    UnloadTexture(preview_texture);
    UnloadRenderTexture(target);
    net.save("data/best_model.bin");
    return 0;
}
