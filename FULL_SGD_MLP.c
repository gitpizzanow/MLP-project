#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define LAYERS 4
#define EPOCHS 100
#define Lr 0.1
#define EPSILON 1e-7f
#define ROWS 3
#define COLS 2
#define BATCH_SIZE 1

typedef struct Layer {
  int input_size;
  int output_size;

  float* weights;  // [input_size * output_size]
  float* biases;   // [output_size]
  float* deltas;   // [output_size]
  float* outputs;  // [output_size]

  float* dW;
  float* dB;

  struct Layer* next;
} Layer;

// ----- Function Prototypes
float sigmoid(float a);
int step(float a);
float derivative(float a);
float cross(float y, float Yhat);
void free_layer(Layer* layer);
Layer create(int input_size, int output_size);
void forward(float* input, Layer* layer);
void compute_deltas(float y, Layer** layers, int num_layers);
void accumulate_gradients(float* input, Layer* layer);
void apply_gradients(Layer* layer, float lr, int batch_size);
// ----- Activation Functions
float sigmoid(float a) { return 1.0f / (1 + exp(-a)); }
int step(float a) { return (a >= 0.5f) ? 1 : 0; }
float derivative(float a) { return a * (1 - a); }
float cross(float y, float Yhat) {
  Yhat = fmaxf(Yhat, EPSILON);
  Yhat = fminf(Yhat, 1.0f - EPSILON);
  return -(y * logf(Yhat) + (1 - y) * logf(1.0f - Yhat));
}

// ----- Memory Functions
void free_layer(Layer* layer) {
  free(layer->weights);
  free(layer->biases);
  free(layer->outputs);
  free(layer->deltas);
  free(layer->dW);
  free(layer->dB);
}

// ----- Layer Creation
Layer create(int input_size, int output_size) {
  Layer layer;
  layer.input_size = input_size;
  layer.output_size = output_size;

  layer.weights = (float*)calloc(input_size * output_size, sizeof(float));
  layer.biases = (float*)calloc(output_size, sizeof(float));
  layer.outputs = (float*)calloc(output_size, sizeof(float));
  layer.deltas = (float*)calloc(output_size, sizeof(float));
  layer.dW = (float*)calloc(input_size * output_size, sizeof(float));
  layer.dB = (float*)calloc(output_size, sizeof(float));
  layer.next = NULL;

  for (int i = 0; i < input_size * output_size; i++)
    layer.weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2;

  for (int i = 0; i < output_size; i++) {
    layer.biases[i] = ((float)rand() / RAND_MAX - 0.5f) * 2;
    layer.outputs[i] = 0;
    layer.deltas[i] = 0;
    layer.dW[i] = 0;
    layer.dB[i] = 0;
  }

  return layer;
}

// ----- Forward Pass
void forward(float* input, Layer* layer) {
  for (int j = 0; j < layer->output_size; j++) {
    float z = 0.0f;
    for (int i = 0; i < layer->input_size; i++)
      z += input[i] * layer->weights[j * layer->input_size + i];
    layer->outputs[j] = sigmoid(z + layer->biases[j]);
  }
}

void compute_deltas(float y, Layer** layers, int num_layers) {
  Layer* out = layers[num_layers - 1];
  float yhat = *(out->outputs);
  *(out->deltas) = (yhat - y) * derivative(yhat);

  for (int l = num_layers - 2; l >= 0; l--) {
    Layer* current = layers[l];  // l :layer
    Layer* next = layers[l + 1];

    for (int j = 0; j < current->output_size; j++) {
      float sum = 0.0f;
      for (int i = 0; i < next->output_size; i++) {
        int idx = i * next->input_size + j;
        sum += next->weights[idx] * next->deltas[i];
      }

      float a = current->outputs[j];
      current->deltas[j] = sum * derivative(a);
    }
  }
}

void accumulate_gradients(float* input, Layer* layer) {
  for (int j = 0; j < layer->output_size; j++) {
    for (int i = 0; i < layer->input_size; i++) {
      int idx = j * layer->input_size + i;
      layer->dW[idx] += layer->deltas[j] * input[i];
    }
    layer->dB[j] += layer->deltas[j];
  }
}

void apply_gradients(Layer* layer, float lr, int batch_size) {
  for (int j = 0; j < layer->output_size; j++) {
    for (int i = 0; i < layer->input_size; i++) {
      int idx = j * layer->input_size + i;
      layer->weights[idx] -= lr * layer->dW[idx] / batch_size;
      layer->dW[idx] = 0;
    }
    layer->biases[j] -= lr * layer->dB[j] / batch_size;
    layer->dB[j] = 0;
  }
}

void train_sgd(float X[][COLS], float* y, Layer** layers, int num_layers,
               float lr) {
  for (int e = 0; e < EPOCHS; e++) {
    float loss = 0.0f;

    for (int i = 0; i < ROWS; i++) {
      float* input = X[i];

      for (int l = 0; l < num_layers; l++) {
        forward(input, layers[l]);
        input = layers[l]->outputs;
      }
      // loss
      float yhat = *(layers[num_layers - 1]->outputs);
      loss += cross(y[i], yhat);

      // Backward
      compute_deltas(y[i], layers, num_layers);

      // Accumulate
      input = X[i];
      for (int l = 0; l < num_layers; l++) {
        accumulate_gradients(input, layers[l]);
        input = layers[l]->outputs;
      }

      // Apply Gradients (SGD = per sample)
      for (int l = 0; l < num_layers; l++)
        apply_gradients(layers[l], lr, BATCH_SIZE);
    }
    loss /= ROWS;
    printf("Epoch %d | Loss = %.4f\n", e, loss);
  }
}

// ----- Main Program
int main(void) {
  srand(time(NULL));

  Layer hidden1 = create(COLS, 4);
  Layer hidden2 = create(4, 4);
  Layer hidden3 = create(4, 2);
  Layer output = create(2, 1);

  hidden1.next = &hidden2;
  hidden2.next = &hidden3;
  hidden3.next = &output;

  Layer* layers[LAYERS] = {&hidden1, &hidden2, &hidden3, &output};
  float pred[ROWS];

  float X[ROWS][COLS] = {{0, 0}, {0, 1}, {1, 0}};
  float y[] = {0, 1, 1};

  train_sgd(X, y, layers, LAYERS, 0.5);

  printf("\n--- Evaluation ---\n");
  int correct = 0;
  for (int i = 0; i < ROWS; i++) {
    float* input = X[i];
    for (int l = 0; l < LAYERS; l++) {
      forward(input, layers[l]);
      input = layers[l]->outputs;
    }
    float yhat = *(layers[LAYERS - 1]->outputs);
    int pred = step(yhat);
    if (pred == (int)y[i]) correct++;
  }

  float acc = 100.0f * correct / ROWS;
  printf("Accuracy: %.2f%% (%d/%d correct)\n", acc, correct, ROWS);

  for (int l = 0; l < LAYERS; l++) free_layer(layers[l]);

  return 0;
}
