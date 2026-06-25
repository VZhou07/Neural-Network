#include <iostream>
#include <raylib.h>
#include "neural_net.hpp"
#include <thread>
#include <Eigen/Dense>
#include <atomic>

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
    while (!WindowShouldClose()){
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
            ImageResizeNN(&image, 28, 28);
            Image display=ImageCopy(image);
            Eigen::MatrixXd input=Eigen::MatrixXd::Zero(784, 1);

            //size display back up to see the data loss
            //ImageResizeNN(&display, 280, 280);
            
            const unsigned char* image_data=static_cast<const unsigned char*>(image.data); //gives rgba
            for(int i=0;i<28;i++){
                for(int j=0;j<28;j++){
                    int index=(i*28+j)*4;
                    input(i*28+j,0)=image_data[index]/255.0;
                }
            }
            std::cout << "Input: " << input << std::endl;
            prediction=net.test(input);
            std::cout << "Prediction: " << prediction << std::endl;
            if (has_preview){
                UnloadTexture(preview_texture);
            }
            preview_texture=LoadTextureFromImage(display);
            has_preview=true;
            UnloadImage(image);
            UnloadImage(display);
            }
            
        //DRAW CANVAS OF 560x560 filled with white
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
    net.save("data/best_model.bin");
    return 0;
}
