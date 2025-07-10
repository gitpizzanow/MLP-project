#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define ROWS 557
#define COLS 784  // 28x28
#define N 4
#define EPSILON 1e-7f

/*
 DATA : [https://www.kaggle.com/c/dogs-vs-cats/data]
*/

float X[ROWS][COLS];
float y[ROWS];

void load_data() {
  FILE* fx = fopen("X_data.txt", "r");
  FILE* fy = fopen("y_data.txt", "r");

  if (!fx || !fy) {
    perror("Failed to open data files.");
    exit(1);
  }

  for (int i = 0; i < ROWS; i++)
    for (int j = 0; j < COLS; j++) fscanf(fx, "%f,", &X[i][j]);

  for (int i = 0; i < ROWS; i++) fscanf(fy, "%f,", &y[i]);

  fclose(fx);
  fclose(fy);
}



float sigmoid(float a) { return 1.0f / (1 + exp(-a)); }

int step(float a) { return (a >= 0.5) ? 1 : 0; }

float cross(float y, float Yhat) {
  Yhat = fmaxf(Yhat, EPSILON);
  Yhat = fminf(Yhat, 1.0f - EPSILON);
  return -(y * logf(Yhat) + (1 - y) * logf(1.0f - Yhat));
}

void backward(float* X, float* weights, float* bayes, float* dv, float* db,
              float* H, float* pred, const float* y, int k, float lr) {
  float dh[N];
  float error = pred[k] - y[k];

  // use dv to compute dh
  for (size_t i = 0; i < N; i++) dh[i] = error * dv[i] * H[i] * (1.0f - H[i]);

  // update dv and db
  for (size_t i = 0; i < N; i++) dv[i] -= lr * error * H[i];

  (*db) -= lr * error;

  for (size_t j = 0; j < N; j++) {
    for (size_t i = 0; i < COLS; i++) {
      weights[j * COLS + i] -= lr * dh[j] * X[k * COLS + i];
    }
    bayes[j] -= lr * dh[j];
  }
}

void forward(float* X, float* weights, float* bayes, float* dv, float* db,
             float* H, float* pred, const float* y, int k) {
  for (size_t j = 0; j < N; j++) {
    float Z = 0;
    for (size_t i = 0; i < COLS; i++)
      Z += X[k * COLS + i] * weights[j * COLS + i];

    H[j] = sigmoid(Z + bayes[j]);
    // printf("Z%zu = %.2f \n",j ,H[j]);
  }

  float yhat = 0, predict = 0;
  for (size_t j = 0; j < N; j++) yhat += (H[j] * dv[j]);

  predict = sigmoid(yhat + *db);
  pred[k] = predict;
  // printf("Y%d = %.0f , pred = %.2f , ŷ=%d\n", k, y[k], predict,
  // step(predict));
  // printf("Y%d = %.0f , ŷ=%d\n", k, y[k], step(predict));
}

int main() {
    
  load_data();

  int train_size = 500;
  int val_size = ROWS - train_size;
  float H[10];  // one hidden layer
  float weights[N * COLS];
  // w11, w21 -> h1 | w12, w22 -> h2 | w13, w33 -> h3
  float bayes[N];
  float dv[N];
  float db;

  srand(time(NULL));
  for (size_t i = 0; i < N * COLS; i++)
    weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2;
  for (size_t i = 0; i < N; i++) {
    bayes[i] = ((float)rand() / RAND_MAX - 0.5f) * 2;
    dv[i] = ((float)rand() / RAND_MAX - 0.5f) * 2;
  }
  db = ((float)rand() / RAND_MAX - 0.5f) * 2;
  float pred[ROWS];

  float lr = 0.1;
  size_t epochs = 100;

  // training

  for (size_t e = 0; e < epochs; e++) {
    int correct = 0;
    float loss = 0.0f;
    printf("epoch°%zu :\n", e);
    for (size_t i = 0; i < train_size; i++) {
      forward((float*)X, weights, bayes, dv, &db, H, pred, y, i);

      loss += cross(y[i], pred[i]);
      if ((int)y[i] == step(pred[i])) correct++;
      backward((float*)X, weights, bayes, dv, &db, H, pred, y, i, lr);
    }

    printf("Loss: %.4f | Accuracy: %.2f%%\n", loss / ROWS,
           (100.0f * correct) / ROWS);
    if (correct == ROWS) {
      printf("Stopping at epoch %zu.\n", e);
      break;
    }
  }

  /*
  If training accuracy is high but validation accuracy is low : Overfitting
  */
  printf("\nEvaluating on validation set:\n");
  int val_correct = 0;
  float val_loss = 0.0f;
  for (size_t i = train_size; i < ROWS; i++) {
    forward((float*)X, weights, bayes, dv, &db, H, pred, y, i);
    val_loss += cross(y[i], pred[i]);
    if ((int)y[i] == step(pred[i])) val_correct++;
  }
  printf("Val Loss: %.4f | Val Accuracy: %.2f%%\n", val_loss / val_size,
         (100.0f * val_correct) / val_size);
}