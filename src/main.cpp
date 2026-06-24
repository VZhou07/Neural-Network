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
        float learning_rate=0.01;
        net.train(learning_rate);
    }
    float accuracy=net.test_accuracy();
    std::cout << "Accuracy: " << accuracy << std::endl;
    done_training=true;
}

int main() {
    NeuralNet net({784,10, 10}, 32,false,true);
    const int width=1280;
    const int height=720;
    InitWindow(width, height, "Number Classifier");
    std::thread t(training_thread, std::ref(net));
    float brush_size=10.0;
    SetTargetFPS(60);
    Rectangle draw_rectangle={10, 30, 560, 560};
    RenderTexture2D target=LoadRenderTexture(560, 560);
    BeginTextureMode(target);
    ClearBackground(BLACK);
    EndTextureMode();
    Texture2D preview_texture={0,0,0,0};
    bool has_preview=false;
    while (!WindowShouldClose()){
        Vector2 mouse_position=GetMousePosition();

        if (CheckCollisionPointRec(mouse_position, draw_rectangle)){
            if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)){
                Vector2 local_mouse_position={mouse_position.x-draw_rectangle.x, mouse_position.y-draw_rectangle.y};
                BeginTextureMode(target);
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
            CloseWindow();
            return 0;
        }

        if (IsKeyPressed(KEY_ENTER)){

            if (!done_training){
                DrawText("Training not done, please wait", 200, 10, 20, BLACK);
                continue;
            }
            Image image=LoadImageFromTexture(target.texture);
            ImageResizeNN(&image, 28, 28);
            Eigen::MatrixXd input=Eigen::MatrixXd::Zero(784, 1);
            const unsigned char* image_data=static_cast<const unsigned char*>(image.data); //gives rgba
            for(int i=0;i<28;i++){
                for(int j=0;j<28;j++){
                    int index=(i*28+j)*4;
                    input(i*28+j,0)=image_data[index]/255.0;
                }
            }
            int prediction=net.test(input);
            std::cout << "Prediction: " << prediction << std::endl;
            if (has_preview){
                UnloadTexture(preview_texture);
            }
            preview_texture=LoadTextureFromImage(image);
            has_preview=true;
            UnloadImage(image);
            if (prediction){
                BeginDrawing();
                //write prediction under preview texture
                std::string prediction_string="prediction: "+std::to_string(prediction);
                DrawText(prediction_string.c_str(), 700, 30+preview_texture.height+10, 20, BLACK);
                EndDrawing();
            }


        }

        //DRAW CANVAS OF 560x560 filled with white
        BeginDrawing();
        ClearBackground(GRAY);
        DrawRectangleLinesEx(draw_rectangle, 2, BLACK);
        DrawTextureRec(target.texture, {0, 0, 560, -560}, {10, 30}, WHITE);
        DrawText("Click Enter to Submit Drawing", 200, 10, 20, BLACK);
        if (has_preview){
            DrawTexture(preview_texture, 700, 30, WHITE);
        }
        EndDrawing();
    }
    net.save("data/best_model.bin");
    t.join();
    CloseWindow();
    return 0;
}
