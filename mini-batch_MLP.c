#include <stdio.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#define ROWS 3
#define COLS 2  
#define N 4
#define BATCH 32
#define EPOCHS 100
#define Lr 0.1
#define EPSILON 1e-7f

float sigmoid(float a) { return 1.0f / (1 + exp(-a)); }

int step(float a) { return (a >= 0.5) ? 1 : 0; }

float derivative(float a){
  return a * (1-a);
}


float cross(float y, float Yhat) {
  Yhat = fmaxf(Yhat, EPSILON);
  Yhat = fminf(Yhat, 1.0f - EPSILON);
  return -(y * logf(Yhat) + (1 - y) * logf(1.0f - Yhat));
}


void accumulate(float* X, float* y , float* pred,
  float* dv,float* H,float* v,
  float* dw,float* dbh,int k,float* db
){
  float error = pred[k] - y[k];
  
  for(int i=0;i<N;i++){
    dv[i]+= error*H[i];
    float dh = error* v[i]*derivative(H[i]);
    
    
    for(int j=0;j<COLS;j++){
      dw[i * COLS + j]+= dh * X[k * COLS + j];
    }
    dbh[i]+=dh;
  }
  (*db)+=error;
}

void apply_gradients(float* v ,float* dv, int batch_size,
float* b,float* db,float* weights,float* dw,
float* dbh,float* bayes
){
  for(int i=0;i<N;i++){
    v[i]-= Lr * dv[i]/batch_size;
  }
  *b -= Lr * (*db)/batch_size;
  
  
   for (int i = 0; i < N; i++) {
    for (int j = 0; j < COLS; j++)
    weights[i * COLS + j] -= Lr * dw[i * COLS + j] / batch_size;
    
    bayes[i] -=Lr* dbh[i] / batch_size;
   }
}

void forward(float* X, float* weights,
    float* bayes, float* v, float* b,
    float* H, float* pred, const float* y, int k) {
    
     for (size_t j = 0; j < N; j++) {
     float Z = 0;
     for (size_t i = 0; i < COLS; i++)
     Z += X[k * COLS + i] * weights[j * COLS + i];

     H[j] = sigmoid(Z + bayes[j]);
  }

  float yhat = 0, predict = 0;
  for (size_t j = 0; j < N; j++) yhat += (H[j] * v[j]);

  predict = sigmoid(yhat + *b);
  pred[k] = predict;
}

int main(){
  srand(time(NULL));
  
  float X[ROWS][COLS]= {
    {1,1},
    {0,0},
    {1,0}
  };
  
  float y[ROWS] = {0,0,1};
  
  
  float H[N];  // one hidden layer
  float weights[N * COLS]; //weights of input to hidden
  // w11, w21 -> h1 | w12, w22 -> h2 | w13, w33 -> h3
  float bayes[N]; // bayes of input to hidden
  float v[N]; //weights of hidden to output
  float b; // bayes of hidden to output
  
  
  float dv[N], dw[N * COLS], dbh[N];
  float db;
  
  for (size_t i = 0; i < N * COLS; i++)
    weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2;
  for (size_t i = 0; i < N; i++) {
    bayes[i] = ((float)rand() / RAND_MAX - 0.5f) * 2;
    v[i] = ((float)rand() / RAND_MAX - 0.5f) * 2;
  }
  b = ((float)rand() / RAND_MAX - 0.5f) * 2;
  float pred[ROWS];
  
  
    
    
  //Training 
  for (size_t e = 0; e < EPOCHS; e++) {
    int correct = 0;
    float loss = 0.0f;
    printf("epoch°%zu :\n", e);
    
    for(int batch_start=0;
    batch_start<ROWS;batch_start+=BATCH){
      
      int batch_end = (batch_start+BATCH>ROWS)?ROWS:batch_start+BATCH;
      int batch_size = batch_end- batch_start;
      
      for (int i = 0; i < N; i++) {
      dv[i] = 0;
      dbh[i] = 0;
      for (int j = 0; j < COLS; j++)
        dw[i * COLS + j] = 0;
    }
    db = 0;
      
      
       for (size_t k = batch_start; k < batch_end; k++) {
      forward((float*)X, weights, bayes, v, &b, H, pred, y, k);
      loss += cross(y[k], pred[k]);
      //backward
      accumulate((float*)X,y,pred,dv,H,v,dw,dbh,k,&db); 
 
      if ((int)y[k] == step(pred[k])) correct++;
 
    }
     apply_gradients(v, dv, batch_size, &b, &db, weights, dw, dbh, bayes);
    }
    
    printf("Loss: %.4f | Accuracy: %.2f%%\n", loss / ROWS,
           (100.0f * correct) / ROWS);
    if (correct == ROWS) {
      printf("Stopping at epoch %zu.\n", e);
      break;
    }
  }
    
    
    
    
    
    
    
    
    
}
