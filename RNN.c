#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define VOCAB_SIZE 128 
#define HIDDEN_SIZE 100
#define SEQUENCE_LENGTH 25
#define LEARNING_RATE 0.01

typedef struct {

    float *wx[VOCAB_SIZE];

    float *wh[HIDDEN_SIZE];
    float *wy[HIDDEN_SIZE];
    float *hidden;
    float *output;
} Model;

Model* init_model() {
    Model *model = (Model*)malloc(sizeof(Model));
    
    for (int i = 0; i < VOCAB_SIZE; i++) {
        model->wx[i] = (float*)malloc(HIDDEN_SIZE * sizeof(float));
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            model->wx[i][j] = ((float)rand() / RAND_MAX * 2 - 1) * 0.01;
        }
    }

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        model->wh[i] = (float*)malloc(HIDDEN_SIZE * sizeof(float));
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            model->wh[i][j] = ((float)rand() / RAND_MAX * 2 - 1) * 0.01;
        }
    }

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        model->wy[i] = (float*)malloc(VOCAB_SIZE * sizeof(float));
        for (int j = 0; j < VOCAB_SIZE; j++) {
            model->wy[i][j] = ((float)rand() / RAND_MAX * 2 - 1) * 0.01;
        }
    }

    model->hidden = (float*)calloc(HIDDEN_SIZE, sizeof(float));
    model->output = (float*)calloc(VOCAB_SIZE, sizeof(float));
    
    return model;
}

float tanh_activation(float x) {
    return tanh(x);
}

void softmax(float* input, int length) {
    float max = input[0];
    for (int i = 1; i < length; i++) {
        if (input[i] > max) max = input[i];
    }
    
    float sum = 0.0;
    for (int i = 0; i < length; i++) {
        input[i] = exp(input[i] - max);
        sum += input[i];
    }
    
    for (int i = 0; i < length; i++) {
        input[i] /= sum;
    }
}

void forward(Model* model, int input_char) {
    float new_hidden[HIDDEN_SIZE] = {0};

    for (int h = 0; h < HIDDEN_SIZE; h++) {
        new_hidden[h] += model->wx[input_char][h];
    }

    for (int h = 0; h < HIDDEN_SIZE; h++) {
        for (int h2 = 0; h2 < HIDDEN_SIZE; h2++) {
            new_hidden[h] += model->hidden[h2] * model->wh[h2][h];
        }
        new_hidden[h] = tanh_activation(new_hidden[h]);
    }

    memcpy(model->hidden, new_hidden, HIDDEN_SIZE * sizeof(float));
    
    for (int v = 0; v < VOCAB_SIZE; v++) {
        float sum = 0.0;
        for (int h = 0; h < HIDDEN_SIZE; h++) {
            sum += model->hidden[h] * model->wy[h][v];
        }
        model->output[v] = sum;
    }

    softmax(model->output, VOCAB_SIZE);
}

float train_step(Model* model, const char* sequence, int sequence_length) {

    float loss = 0;

    memset(model->hidden, 0, HIDDEN_SIZE * sizeof(float));

    for (int t = 0; t < sequence_length - 1; t++) {
        int input_char = (unsigned char)sequence[t];
        int target_char = (unsigned char)sequence[t + 1];

        forward(model, input_char);

        loss -= log(model->output[target_char] + 1e-10);
        float d_output[VOCAB_SIZE] = {0};
        memcpy(d_output, model->output, VOCAB_SIZE * sizeof(float));
        d_output[target_char] -= 1.0;
        
        for (int h = 0; h < HIDDEN_SIZE; h++) {
            for (int v = 0; v < VOCAB_SIZE; v++) {
                model->wy[h][v] -= LEARNING_RATE * d_output[v] * model->hidden[h];
            }
        }
    }
    
    return loss / (sequence_length - 1);
}


void main(){
    printf("%d",RAND_MAX);
}
void _main(){
    FILE *file;
    char filename[] = "input.txt";  
    char *content;                   
    long fileSize;                   

    file = fopen(filename, "rb");
    if (file == NULL) {
        perror("Error opening file");
        return 1;
    }

    fseek(file, 0, SEEK_END);
    fileSize = ftell(file);          
    rewind(file);                   

    content = (char *)malloc(fileSize + 1); 
    if (content == NULL) {
        perror("Memory allocation failed");
        fclose(file);
        return 1;
    }

    fread(content, 1, fileSize, file);
    content[fileSize] = '\0';        


    Model* model = init_model();
    
    const char* training_text = content;
    int text_length = strlen(training_text);
    
    for (int epoch = 0; epoch < 100; epoch++) {
        float loss = train_step(model, training_text, text_length);
        if (epoch % 10 == 0) {
            printf("Epoch %d: loss = %f\n", epoch, loss);
        }
    }
    
    
    printf("\nGenerated text: ");
    char current_char = 'H';
    for (int i = 0; i < 20; i++) {
        printf("%c", current_char);
        forward(model, (unsigned char)current_char);
        
        float r = (float)rand() / RAND_MAX;
        float cum_prob = 0.0;
        for (int j = 0; j < VOCAB_SIZE; j++) {
            cum_prob += model->output[j];
            if (r < cum_prob) {
                current_char = (char)j;
                break;
            }
        }
    }
    printf("\n");
    
}

