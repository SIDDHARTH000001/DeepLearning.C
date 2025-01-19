#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define SEQUENCE_LENGTH 32
#define EMBEDDING_DIM 64
#define HEAD_DIM 32
#define NUM_HEADS 2
#define SQRT_HEAD_DIM 5.656854f  // sqrt(32)


typedef struct {
    // Attention weights for each head
    float *W_query[NUM_HEADS];  // [EMBEDDING_DIM x HEAD_DIM]
    float *W_key[NUM_HEADS];    // [EMBEDDING_DIM x HEAD_DIM]
    float *W_value[NUM_HEADS];  // [EMBEDDING_DIM x HEAD_DIM]
    float *W_output;            // [NUM_HEADS * HEAD_DIM x EMBEDDING_DIM]
    
    // Temporary storage for attention computation
    float *queries[NUM_HEADS];     // [SEQUENCE_LENGTH x HEAD_DIM]
    float *keys[NUM_HEADS];        // [SEQUENCE_LENGTH x HEAD_DIM]
    float *values[NUM_HEADS];      // [SEQUENCE_LENGTH x HEAD_DIM]
    float *attention_scores[NUM_HEADS];  // [SEQUENCE_LENGTH x SEQUENCE_LENGTH]
    float *attention_output;       // [SEQUENCE_LENGTH x EMBEDDING_DIM]
} MultiHeadAttention;


MultiHeadAttention* init_attention() {
    MultiHeadAttention *attn = (MultiHeadAttention*)malloc(sizeof(MultiHeadAttention));
    
    for (int h = 0; h < NUM_HEADS; h++) {
        attn->W_query[h] = (float*)malloc(EMBEDDING_DIM * HEAD_DIM * sizeof(float));
        attn->W_key[h] = (float*)malloc(EMBEDDING_DIM * HEAD_DIM * sizeof(float));
        attn->W_value[h] = (float*)malloc(EMBEDDING_DIM * HEAD_DIM * sizeof(float));
        
        // Initialize with small random values
        for (int i = 0; i < EMBEDDING_DIM * HEAD_DIM; i++) {
            attn->W_query[h][i] = ((float)rand() / RAND_MAX * 2 - 1) * 0.02;
            attn->W_key[h][i] = ((float)rand() / RAND_MAX * 2 - 1) * 0.02;
            attn->W_value[h][i] = ((float)rand() / RAND_MAX * 2 - 1) * 0.02;
        }
    }

    // Initialize output projection
    attn->W_output = (float*)malloc(NUM_HEADS * HEAD_DIM * EMBEDDING_DIM * sizeof(float));
    for (int i = 0; i < NUM_HEADS * HEAD_DIM * EMBEDDING_DIM; i++) {
        attn->W_output[i] = ((float)rand() / RAND_MAX * 2 - 1) * 0.02;
    }

    for (int h = 0; h < NUM_HEADS; h++) {
        attn->queries[h] = (float*)malloc(SEQUENCE_LENGTH * HEAD_DIM * sizeof(float));
        attn->keys[h] = (float*)malloc(SEQUENCE_LENGTH * HEAD_DIM * sizeof(float));
        attn->values[h] = (float*)malloc(SEQUENCE_LENGTH * HEAD_DIM * sizeof(float));
        attn->attention_scores[h] = (float*)malloc(SEQUENCE_LENGTH * SEQUENCE_LENGTH * sizeof(float));
    }
    attn->attention_output = (float*)malloc(SEQUENCE_LENGTH * EMBEDDING_DIM * sizeof(float));
    
    return attn;
}

void matrix_multiply(float* A, float* B, float* C, 
                    int A_rows, int A_cols, int B_cols) {
    for (int i = 0; i < A_rows; i++) {
        for (int j = 0; j < B_cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < A_cols; k++) {
                sum += A[i * A_cols + k] * B[k * B_cols + j];
            }
            C[i * B_cols + j] = sum;
        }
    }
}

void softmax_rows(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        // Find max for numerical stability
        float max_val = matrix[i * cols];
        for (int j = 1; j < cols; j++) {
            if (matrix[i * cols + j] > max_val) {
                max_val = matrix[i * cols + j];
            }
        }
        
        // Compute exp and sum
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] = exp(matrix[i * cols + j] - max_val);
            sum += matrix[i * cols + j];
        }
        
        // Normalize
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] /= sum;
        }
    }
}



void self_attention(MultiHeadAttention* attn, float* input_sequence, int seq_length) {
    // For each attention head
    for (int h = 0; h < NUM_HEADS; h++) {
        // Compute Q, K, V for this head
        matrix_multiply(input_sequence, attn->W_query[h], 
                       attn->queries[h], seq_length, EMBEDDING_DIM, HEAD_DIM);
        matrix_multiply(input_sequence, attn->W_key[h], 
                       attn->keys[h], seq_length, EMBEDDING_DIM, HEAD_DIM);
        matrix_multiply(input_sequence, attn->W_value[h], 
                       attn->values[h], seq_length, EMBEDDING_DIM, HEAD_DIM);
        
        // Compute attention scores (Q * K^T / sqrt(d_k))
        matrix_multiply(attn->queries[h], attn->keys[h], 
                       attn->attention_scores[h], seq_length, HEAD_DIM, seq_length);
        
        // Scale attention scores
        for (int i = 0; i < seq_length * seq_length; i++) {
            attn->attention_scores[h][i] /= SQRT_HEAD_DIM;
        }
        
        // Apply softmax
        softmax_rows(attn->attention_scores[h], seq_length, seq_length);
        
        // Compute attention output
        float* head_output = (float*)malloc(seq_length * HEAD_DIM * sizeof(float));
        matrix_multiply(attn->attention_scores[h], attn->values[h], 
                       head_output, seq_length, seq_length, HEAD_DIM);
        
        // Copy to concatenated output
        for (int i = 0; i < seq_length; i++) {
            for (int j = 0; j < HEAD_DIM; j++) {
                attn->attention_output[i * (NUM_HEADS * HEAD_DIM) + h * HEAD_DIM + j] = 
                    head_output[i * HEAD_DIM + j];
            }
        }
        
        free(head_output);
    }
    
    // Final output projection
    float* final_output = (float*)malloc(seq_length * EMBEDDING_DIM * sizeof(float));
    matrix_multiply(attn->attention_output, attn->W_output, 
                   final_output, seq_length, NUM_HEADS * HEAD_DIM, EMBEDDING_DIM);
    
    // Copy result back to attention_output
    memcpy(attn->attention_output, final_output, 
           seq_length * EMBEDDING_DIM * sizeof(float));
    
    free(final_output);
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define VOCAB_SIZE 256  // For ASCII characters
#define SEQUENCE_LENGTH 32
#define EMBEDDING_DIM 64
#define BATCH_SIZE 16
#define LEARNING_RATE 0.001f


typedef struct {
    float* weight_matrix;  // [VOCAB_SIZE x EMBEDDING_DIM]
    float* gradients;      // Same size as weight_matrix
} Embedding;


typedef struct {
    Embedding* embedding;
    MultiHeadAttention* attention;
    float* output_layer;   // [EMBEDDING_DIM x VOCAB_SIZE]
    float* output_gradients;
    float* embedding_gradients;
} TrainingContext;


Embedding* init_embedding() {
    Embedding* emb = (Embedding*)malloc(sizeof(Embedding));
    
    emb->weight_matrix = (float*)malloc(VOCAB_SIZE * EMBEDDING_DIM * sizeof(float));
    emb->gradients = (float*)calloc(VOCAB_SIZE * EMBEDDING_DIM, sizeof(float));
    
    for (int i = 0; i < VOCAB_SIZE * EMBEDDING_DIM; i++) {
        emb->weight_matrix[i] = ((float)rand() / RAND_MAX * 2 - 1) * 0.02f;
    }
    
    return emb;
}


TrainingContext* init_training_context() {
    TrainingContext* ctx = (TrainingContext*)malloc(sizeof(TrainingContext));
    
    ctx->embedding = init_embedding();
    ctx->attention = init_attention();  
    
    ctx->output_layer = (float*)malloc(EMBEDDING_DIM * VOCAB_SIZE * sizeof(float));
    ctx->output_gradients = (float*)calloc(EMBEDDING_DIM * VOCAB_SIZE, sizeof(float));
    ctx->embedding_gradients = (float*)calloc(SEQUENCE_LENGTH * EMBEDDING_DIM, sizeof(float));
    
    for (int i = 0; i < EMBEDDING_DIM * VOCAB_SIZE; i++) {
        ctx->output_layer[i] = ((float)rand() / RAND_MAX * 2 - 1) * 0.02f;
    }
    
    return ctx;
}


void prepare_batch(const char* text, int text_length, int batch_idx,
                  float* input_batch, float* target_batch) {
    int start_idx = (batch_idx * SEQUENCE_LENGTH) % (text_length - SEQUENCE_LENGTH);
    
    memset(input_batch, 0, SEQUENCE_LENGTH * VOCAB_SIZE * sizeof(float));
    memset(target_batch, 0, SEQUENCE_LENGTH * VOCAB_SIZE * sizeof(float));
    
    for (int i = 0; i < SEQUENCE_LENGTH; i++) {
        int input_char = (unsigned char)text[start_idx + i];
        int target_char = (unsigned char)text[start_idx + i + 1];
        
        input_batch[i * VOCAB_SIZE + input_char] = 1.0f;
        target_batch[i * VOCAB_SIZE + target_char] = 1.0f;
    }
}

// Forward pass
float forward_pass(TrainingContext* ctx, float* input_batch, float* embedded_sequence) {
    
    matrix_multiply(input_batch, ctx->embedding->weight_matrix,
                   embedded_sequence, SEQUENCE_LENGTH, VOCAB_SIZE, EMBEDDING_DIM);
    
    
    self_attention(ctx->attention, embedded_sequence, SEQUENCE_LENGTH);
    
    
    float* logits = (float*)malloc(SEQUENCE_LENGTH * VOCAB_SIZE * sizeof(float));
    matrix_multiply(ctx->attention->attention_output, ctx->output_layer,
                   logits, SEQUENCE_LENGTH, EMBEDDING_DIM, VOCAB_SIZE);
    
    float total_loss = 0.0f;
    for (int i = 0; i < SEQUENCE_LENGTH; i++) {
       
        float max_val = logits[i * VOCAB_SIZE];
        for (int j = 1; j < VOCAB_SIZE; j++) {
            if (logits[i * VOCAB_SIZE + j] > max_val) {
                max_val = logits[i * VOCAB_SIZE + j];
            }
        }
        
        float sum = 0.0f;
        for (int j = 0; j < VOCAB_SIZE; j++) {
            logits[i * VOCAB_SIZE + j] = exp(logits[i * VOCAB_SIZE + j] - max_val);
            sum += logits[i * VOCAB_SIZE + j];
        }

        // Normalize logits
        for (int j = 0; j < VOCAB_SIZE; j++) {
            logits[i * VOCAB_SIZE + j] /= sum;
        }

        // Cross-entropy loss (using target_batch instead of input_batch for true targets)
        for (int j = 0; j < VOCAB_SIZE; j++) {
            if (input_batch[(i + 1) * VOCAB_SIZE + j] > 0.8f) {  // target batch should be used here
                total_loss -= log(logits[i * VOCAB_SIZE + j] + 3e-4f);  // Prevent log(0)
            }
        }
    }
    
    free(logits);
    return total_loss / SEQUENCE_LENGTH;
}

// Backward pass (simplified gradient computation)
void backward_pass(TrainingContext* ctx, float* input_batch, float* embedded_sequence) {
    for (int i = 0; i < SEQUENCE_LENGTH * EMBEDDING_DIM; i++) {
        ctx->embedding_gradients[i] = ctx->attention->attention_output[i] * LEARNING_RATE;
    }
    
    for (int i = 0; i < SEQUENCE_LENGTH; i++) {
        for (int j = 0; j < VOCAB_SIZE; j++) {
            if (input_batch[i * VOCAB_SIZE + j] > 0.2f) {
                for (int k = 0; k < EMBEDDING_DIM; k++) {
                    ctx->embedding->weight_matrix[j * EMBEDDING_DIM + k] -= 
                        ctx->embedding_gradients[i * EMBEDDING_DIM + k];
                }
            }
        }
    }
}

void train_and_generate(const char* training_text, const char* prompt, int length_to_generate) {
    TrainingContext* ctx = init_training_context();
    int text_length = strlen(training_text);
    int num_batches = (text_length - SEQUENCE_LENGTH) / SEQUENCE_LENGTH;
    
    // Allocate batch memory
    float* input_batch = (float*)malloc(SEQUENCE_LENGTH * VOCAB_SIZE * sizeof(float));
    float* target_batch = (float*)malloc(SEQUENCE_LENGTH * VOCAB_SIZE * sizeof(float));
    float* embedded_sequence = (float*)malloc(SEQUENCE_LENGTH * EMBEDDING_DIM * sizeof(float));
    
    // Training loop (same as before)
    for (int epoch = 0; epoch < 20; epoch++) {
        float epoch_loss = 0.0f;
        
        for (int batch = 0; batch < num_batches; batch++) {
            prepare_batch(training_text, text_length, batch, input_batch, target_batch);
            float loss = forward_pass(ctx, input_batch, embedded_sequence);
            epoch_loss += loss;
            backward_pass(ctx, input_batch, embedded_sequence);
            
            if (batch % 1000 == 0) {
                printf("Epoch %d, Batch %d/%d, Loss: %f\n", 
                       epoch, batch, num_batches, loss);
            }
        }
        
        printf("Epoch %d complete, Average Loss: %f\n", 
               epoch, epoch_loss / num_batches);
    }
    save_model(ctx, "my_trained_model.bin");
    printf("\nGenerating text with prompt: %s\n", prompt);
    
    memset(input_batch, 0, SEQUENCE_LENGTH * VOCAB_SIZE * sizeof(float));
    int prompt_len = strlen(prompt);
    for (int i = 0; i < prompt_len && i < SEQUENCE_LENGTH; i++) {
        input_batch[i * VOCAB_SIZE + (unsigned char)prompt[i]] = 1.0f;
    }
    
    printf("%s", prompt);
    for (int i = 0; i < length_to_generate; i++) {
        matrix_multiply(input_batch, ctx->embedding->weight_matrix,
                       embedded_sequence, SEQUENCE_LENGTH, VOCAB_SIZE, EMBEDDING_DIM);
        
        self_attention(ctx->attention, embedded_sequence, SEQUENCE_LENGTH);
        
        float* logits = (float*)malloc(VOCAB_SIZE * sizeof(float));
        matrix_multiply(ctx->attention->attention_output + (SEQUENCE_LENGTH - 1) * EMBEDDING_DIM,
                       ctx->output_layer, logits, 1, EMBEDDING_DIM, VOCAB_SIZE);
        
        float max_val = logits[0];
        for (int j = 1; j < VOCAB_SIZE; j++) {
            if (logits[j] > max_val) max_val = logits[j];
        }
        
        float sum = 0.0f;
        for (int j = 0; j < VOCAB_SIZE; j++) {
            logits[j] = exp(logits[j] - max_val);
            sum += logits[j];
        }
        
        float r = (float)rand() / RAND_MAX * sum;
        float cumsum = 0.0f;
        int next_char = 0;
        
        for (int j = 0; j < VOCAB_SIZE; j++) {
            cumsum += logits[j];
            if (r <= cumsum) {
                next_char = j;
                break;
            }
        }
        
        printf("%c", next_char);
        fflush(stdout);
        
        memmove(input_batch, input_batch + VOCAB_SIZE, 
                (SEQUENCE_LENGTH - 1) * VOCAB_SIZE * sizeof(float));
        
        memset(input_batch + (SEQUENCE_LENGTH - 1) * VOCAB_SIZE, 0, VOCAB_SIZE * sizeof(float));
        input_batch[(SEQUENCE_LENGTH - 1) * VOCAB_SIZE + next_char] = 1.0f;
        
        free(logits);
    }
    printf("\n");
    
    free(input_batch);
    free(target_batch);
    free(embedded_sequence);
}

void save_model(TrainingContext* ctx, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error: Could not open file for writing\n");
        return;
    }
    
    fwrite(ctx->embedding->weight_matrix, sizeof(float), 
           VOCAB_SIZE * EMBEDDING_DIM, file);
    
    for (int h = 0; h < NUM_HEADS; h++) {
        fwrite(ctx->attention->W_query[h], sizeof(float), 
               EMBEDDING_DIM * HEAD_DIM, file);
        fwrite(ctx->attention->W_key[h], sizeof(float), 
               EMBEDDING_DIM * HEAD_DIM, file);
        fwrite(ctx->attention->W_value[h], sizeof(float), 
               EMBEDDING_DIM * HEAD_DIM, file);
    }
    
    fwrite(ctx->attention->W_output, sizeof(float), 
           NUM_HEADS * HEAD_DIM * EMBEDDING_DIM, file);
    
    fwrite(ctx->output_layer, sizeof(float), 
           EMBEDDING_DIM * VOCAB_SIZE, file);
    
    fclose(file);
    printf("Model saved to %s\n", filename);
}

TrainingContext* load_model(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Could not open file for reading\n");
        return NULL;
    }
    
    // Initialize a new context
    TrainingContext* ctx = init_training_context();
    
    // Load embedding weights
    if (fread(ctx->embedding->weight_matrix, sizeof(float), 
              VOCAB_SIZE * EMBEDDING_DIM, file) != VOCAB_SIZE * EMBEDDING_DIM) {
        printf("Error reading embedding weights\n");
        return NULL;
    }
    
    // Load attention weights for each head
    for (int h = 0; h < NUM_HEADS; h++) {
        if (fread(ctx->attention->W_query[h], sizeof(float), 
                  EMBEDDING_DIM * HEAD_DIM, file) != EMBEDDING_DIM * HEAD_DIM ||
            fread(ctx->attention->W_key[h], sizeof(float), 
                  EMBEDDING_DIM * HEAD_DIM, file) != EMBEDDING_DIM * HEAD_DIM ||
            fread(ctx->attention->W_value[h], sizeof(float), 
                  EMBEDDING_DIM * HEAD_DIM, file) != EMBEDDING_DIM * HEAD_DIM) {
            printf("Error reading attention weights\n");
            return NULL;
        }
    }
    
    // Load attention output projection
    if (fread(ctx->attention->W_output, sizeof(float), 
              NUM_HEADS * HEAD_DIM * EMBEDDING_DIM, file) != 
              NUM_HEADS * HEAD_DIM * EMBEDDING_DIM) {
        printf("Error reading attention output weights\n");
        return NULL;
    }
    
    // Load output layer weights
    if (fread(ctx->output_layer, sizeof(float), 
              EMBEDDING_DIM * VOCAB_SIZE, file) != EMBEDDING_DIM * VOCAB_SIZE) {
        printf("Error reading output layer weights\n");
        return NULL;
    }
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    return ctx;
}

// Function to generate text using a loaded model
void generate_from_saved_model(const char* model_file, const char* prompt, int length_to_generate) {
    TrainingContext* ctx = load_model(model_file);
    if (!ctx) {
        printf("Failed to load model\n");
        return;
    }
    
    // Use the existing text generation code here
    printf("Generating text with prompt: %s\n", prompt);
    
    printf("\nGenerating text with prompt: %s\n", prompt);
    // Allocate batch memory
    float* input_batch = (float*)malloc(SEQUENCE_LENGTH * VOCAB_SIZE * sizeof(float));
    float* target_batch = (float*)malloc(SEQUENCE_LENGTH * VOCAB_SIZE * sizeof(float));
    float* embedded_sequence = (float*)malloc(SEQUENCE_LENGTH * EMBEDDING_DIM * sizeof(float));
    
    // Initialize input sequence with the prompt
    memset(input_batch, 0, SEQUENCE_LENGTH * VOCAB_SIZE * sizeof(float));
    int prompt_len = strlen(prompt);
    for (int i = 0; i < prompt_len && i < SEQUENCE_LENGTH; i++) {
        input_batch[i * VOCAB_SIZE + (unsigned char)prompt[i]] = 1.0f;
    }
    
    // Generate new text character by character
    printf("%s", prompt);
    for (int i = 0; i < length_to_generate; i++) {
        // Get embeddings
        matrix_multiply(input_batch, ctx->embedding->weight_matrix,
                       embedded_sequence, SEQUENCE_LENGTH, VOCAB_SIZE, EMBEDDING_DIM);
        
        // Run attention
        self_attention(ctx->attention, embedded_sequence, SEQUENCE_LENGTH);
        
        // Get next character probabilities
        float* logits = (float*)malloc(VOCAB_SIZE * sizeof(float));
        matrix_multiply(ctx->attention->attention_output + (SEQUENCE_LENGTH - 1) * EMBEDDING_DIM,
                       ctx->output_layer, logits, 1, EMBEDDING_DIM, VOCAB_SIZE);
        
        // Apply softmax
        float max_val = logits[0];
        for (int j = 1; j < VOCAB_SIZE; j++) {
            if (logits[j] > max_val) max_val = logits[j];
        }
        
        float sum = 0.0f;
        for (int j = 0; j < VOCAB_SIZE; j++) {
            logits[j] = exp(logits[j] - max_val);
            sum += logits[j];
        }
        
        // Sample next character
        float r = (float)rand() / RAND_MAX * sum;
        float cumsum = 0.0f;
        int next_char = 0;
        
        for (int j = 0; j < VOCAB_SIZE; j++) {
            cumsum += logits[j];
            if (r <= cumsum) {
                next_char = j;
                break;
            }
        }
        
        // Print and update input sequence
        printf("%c", next_char);
        fflush(stdout);
        
        // Shift input sequence left
        memmove(input_batch, input_batch + VOCAB_SIZE, 
                (SEQUENCE_LENGTH - 1) * VOCAB_SIZE * sizeof(float));
        
        // Add new character to input
        memset(input_batch + (SEQUENCE_LENGTH - 1) * VOCAB_SIZE, 0, VOCAB_SIZE * sizeof(float));
        input_batch[(SEQUENCE_LENGTH - 1) * VOCAB_SIZE + next_char] = 1.0f;
        
        free(logits);
    }
    printf("\n");
    
}

int main(int argc, char* argv[]) {
    const char* type = argv[1];
    const char* model_file = argv[2];
    const char* prompt = argv[3];
    int length_to_generate = atoi(argv[4]); 

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

     
    
    if(strcmp(type, "train") == 0){
        printf("training.......");
        const char* training_text = content;
        const char* prompt = "This is";
        int length_to_generate = 50;
        train_and_generate(training_text, prompt, length_to_generate);
    }
    else{
        generate_from_saved_model(model_file, prompt, length_to_generate);
    }
    return 0;
}