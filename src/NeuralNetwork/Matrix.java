package NeuralNetwork;

import java.util.Random;

public class Matrix {
    private int rows;
    private int cols;
    double[][] data;
    
    public Matrix(int rows, int cols){
        this.rows = rows;
        this.cols = cols;
        this.data = new double[rows][cols];
    }
    
    public void randomizeMatrix(){
        Random rand = new Random();
        for(int i=0; i < this.rows; i++){
            for(int j=0; j < this.cols; j++)
                //random number between -1 and 1
                this.data[i][j] = (Math.random() * 2) - 1;
        }
    }
    
    //scalar operation - to multiply each value of matrix with a value
    public void multiplyMatrixByValue(double n){
        for(int i=0; i < this.rows ; i++){
            for(int j=0; j < this.cols; j++)
                this.data[i][j] *= n;
        }
    }
    
    //multiplication of 2 matrix - element by element -- SCALAR operation !!
    public void multiplyMatrixByScalar(Matrix m){        
        for(int i = 0; i < this.rows ; i++){
            for(int j = 0; j < this.cols; j++)
                this.data[i][j] *= m.data[i][j];
            }                
    }
    
    public void addValueToMatrix(int n){        
        for(int i=0; i < rows ; i++){
            for(int j=0; j < cols; j++)
                data[i][j] += n;
        }        
    }
    
    //addition of 2 matrix - element by element
    public void add2Matrix(Matrix m){              
        for(int i=0; i < this.rows ; i++){
            for(int j=0; j < this.cols; j++)
                this.data[i][j] += m.data[i][j];
            }                
    }
    
    //multiply 2 matrices by accepting 2 Matrix objects !
    public static Matrix matrixProductByObjects(Matrix m1, Matrix m2){
        Matrix resultMatrix = null;
        double sum = 0;
        //store in a & b for easy calculation
        Matrix a = m1;
        Matrix b = m2;
        //if cols of 1st matrix == rows of 2nd matrix then only we can multiply
        if(a.cols != b.rows)
            System.out.println("Error - Columns of 1st matrix should match the rows of 2nd matrix");
        else{            
            resultMatrix = new Matrix(a.rows, b.cols);            
            for(int i = 0; i < resultMatrix.rows; i++){
                for(int j = 0 ; j < resultMatrix.cols; j++, sum=0){
                   //dot product of values in col
                    for(int k = 0; k < a.cols; k++){
                       sum += a.data[i][k] * b.data[k][j];                       
                   }
                resultMatrix.data[i][j] = sum;
            }
        }
    }
        return resultMatrix;
}
    
    // transpose the matrix
    public static Matrix transposeMatrix(Matrix m){
        Matrix result = new Matrix(m.cols, m.rows);        
        for(int i = 0; i < m.rows; i++){
            for(int j =0; j < m.cols; j++){                
                result.data[j][i] = m.data[i][j];
            }
        }
        return result;
    }
    
    // print the matrix
    public void printMatrix(){
        for(int i = 0; i < this.data.length; i++) {
           for (int j = 0; j < this.data[i].length; j++) {
            System.out.print(this.data[i][j] + " ");
            }
        System.out.println();
        }
    }
    
    
    public static Matrix fromArrayToMatrix(double[] arr){
        Matrix m = new Matrix(arr.length, 1);
        for(int i = 0; i < arr.length; i++){
            m.data[i][0] = arr[i];
        }
        return m;
    }
    
    public double[] convertMatrixToArray(Matrix m){
        int initialArrayIndex = 0;
        int arrayLength = m.rows * m.cols;  //total number of array elements
        double[] outputArray = new double[arrayLength];        
        for(int i = 0; i < m.rows ; i++){
            for(int j = 0; j < m.cols; j++){
                outputArray[initialArrayIndex++] = m.data[i][j];
            }
        }
        return outputArray;
    }
    
    public void activationFunction(){
        Matrix a = this;
        double currentValue;
        //loop through each element and apply sigmoid function for each element
        for(int i = 0; i < a.rows; i++){
            for(int j = 0; j < a.cols; j++){
              currentValue = a.data[i][j];  
              //pass the current value to Sigmoid function and replace current value
              // with value returned by Sigmoid function
              a.data[i][j] = a.sigmoid(currentValue);              
            }
        }
    }
        
    public double sigmoid(double x){
        return 1 / (1 + Math.exp(-x));
    }
    
    // take each element of matrix and convert it into y * (1-y)
    public static Matrix derivativeOfSigmoid(Matrix a){
        // return sigmoid(x) * (1 - sigmoid(x))                
        for(int i = 0; i < a.rows; i++){
            for(int j = 0; j < a.cols; j++){
                double currentValue = a.data[i][j];            
                a.data[i][j] = currentValue * (1 - currentValue);
            }        
        }
        return a;
    }
    
    public static Matrix substract2Matrix(Matrix targetMatrix, Matrix outputMatrix){
        //error matrix will have same number of rows and cols of any 2 incoming matrix paramterts
        Matrix errorMatrix = new Matrix(targetMatrix.rows, targetMatrix.cols);
        if((targetMatrix.rows != outputMatrix.rows) && (targetMatrix.cols != outputMatrix.cols)){
            System.out.println("Substraction Error - Columns of 1st matrix should match the rows of 2nd matrix");
        }            
        else{ 
            for(int i = 0; i < errorMatrix.rows ; i++){
                for(int j = 0; j < errorMatrix.cols; j++)
                    errorMatrix.data[i][j] = targetMatrix.data[i][j] - outputMatrix.data[i][j];
            }
        }    
        return errorMatrix;
    }
}