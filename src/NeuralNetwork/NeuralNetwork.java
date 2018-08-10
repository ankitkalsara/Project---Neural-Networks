package NeuralNetwork;

//**** class for multi-layer perceptron ****
public class NeuralNetwork {    
    private int inputNeurons;
    private int hiddenNeurons;
    private int outputNeurons;
    private Matrix weightsI_H;    //weights between input to hidden neurons
    private Matrix weightsH_O;    //weights between hidden to output neurons
    private Matrix hiddenBias;    //Bias for hidden neurons
    private Matrix outputBias;    //Bias for output neurons
    private double learningRate;
    
    public void NeuralNetwork(){        
    }    
    
    public NeuralNetwork(int inputNeurons, int hiddenNeurons, int outputNeurons){
        this.inputNeurons  = inputNeurons;
        this.hiddenNeurons = hiddenNeurons;
        this.outputNeurons = outputNeurons;
        
        this.weightsI_H = new Matrix(this.hiddenNeurons, this.inputNeurons);
        this.weightsH_O = new Matrix(this.outputNeurons, this.hiddenNeurons);
        
        this.weightsI_H.randomizeMatrix();
        this.weightsH_O.randomizeMatrix();
        
        this.hiddenBias = new Matrix(this.hiddenNeurons, 1); //it will be column vector
        this.outputBias = new Matrix(this.outputNeurons, 1); //it will be column vector
        
        this.hiddenBias.randomizeMatrix();
        this.outputBias.randomizeMatrix();
        
        this.learningRate = 0.0005;
    }
    
    // used to predicting the output
    public double[] feedForward(double[] input_array){        
        //Generating the hidden matrix
        Matrix inputMatrix = Matrix.fromArrayToMatrix(input_array);  //convert input array into input matrix
        
        //multiply input weights with input --> hiddenMatrix
        Matrix hiddenMatrix = Matrix.matrixProductByObjects(this.weightsI_H, inputMatrix);
        
        //add Bias to the hidden matrix
        hiddenMatrix.add2Matrix(this.hiddenBias);
        
        //apply activation function to EACH element of hidden Matrix!        
        hiddenMatrix.activationFunction();
        
        //now multiply output weights with hidden layer ==> outputMatrix
        Matrix outputMatrix = Matrix.matrixProductByObjects(this.weightsH_O, hiddenMatrix);
        
        //add Bias to the output matrix        
        outputMatrix.add2Matrix(this.outputBias);
        
        //apply activation function to EACH element of output Matrix!
        outputMatrix.activationFunction();
        
        return outputMatrix.convertMatrixToArray(outputMatrix);
    }
    
    public double train(double[] input_array, double[] target_array, int index){                
        //Generating the hidden matrix
        Matrix inputMatrix = Matrix.fromArrayToMatrix(input_array);  //convert the input array into input matrix
        
        //multiply input weights with input = hiddenMatrix
        Matrix hiddenMatrix = Matrix.matrixProductByObjects(this.weightsI_H, inputMatrix);
        
        //add Bias to the hidden matrix
        hiddenMatrix.add2Matrix(this.hiddenBias);
        
        //apply activation function to EACH element of hidden Matrix!        
        hiddenMatrix.activationFunction();
        
        //now multiply output weights with hidden layer ==> outputMatrix
        Matrix outputMatrix = Matrix.matrixProductByObjects(this.weightsH_O, hiddenMatrix);
        
        //add Bias to the output matrix                
        outputMatrix.add2Matrix(this.outputBias);
        
        //apply activation function to EACH element of output Matrix!
        outputMatrix.activationFunction();       
        
        Matrix outputMatrix_copy = outputMatrix;
        double[] target_array_copy = target_array;
        
        //convert target array to matrix object
        Matrix targetMatrix = Matrix.fromArrayToMatrix(target_array);
        
        //below is the technique for Back propogation
        //calculate the output error
        //OUTPUT_ERROR_MATRIX = TARGET_MATRIX - OUTPUT_MATRIX
        Matrix outputErrorsMatrix = Matrix.substract2Matrix(targetMatrix, outputMatrix); 
        
        
        //Calculate Gradient
        // let gradient = outputs * (1 - outputs)
        Matrix gradient = Matrix.derivativeOfSigmoid(outputMatrix);
        
        gradient.multiplyMatrixByScalar(outputErrorsMatrix); //multiplication is element by element
        gradient.multiplyMatrixByValue(this.learningRate);       
        
        //Calculate the Deltas                
        Matrix hiddenT = Matrix.transposeMatrix(hiddenMatrix);
        
        //calculate hidden to --> output deltas 
        Matrix deltaweightsH_O = Matrix.matrixProductByObjects(gradient, hiddenT);
        
        // Adjust the weights by Deltas
        this.weightsH_O.add2Matrix(deltaweightsH_O);
        
        // Adjust the Bias by its deltas (which is just the gradients)
        this.outputBias.add2Matrix(gradient);
        
        
        //calculate the hidden layer errors
        //transpose weights of hidden output, then dot product with output errors        
        Matrix weightsH_O_T = Matrix.transposeMatrix(this.weightsH_O);
        Matrix hidden_errors = Matrix.matrixProductByObjects(weightsH_O_T, outputErrorsMatrix);
        
        //calculate hidden gradient
        Matrix hidden_gradient = Matrix.derivativeOfSigmoid(hiddenMatrix);
        
        hidden_gradient.multiplyMatrixByScalar(hidden_errors); //multiplication is element by element        
        hidden_gradient.multiplyMatrixByValue(this.learningRate);
        
        Matrix inputTranspose = Matrix.transposeMatrix(inputMatrix);
        
        //Calculate input --> hidden deltas 
        Matrix deltaweightsI_H = Matrix.matrixProductByObjects(hidden_gradient, inputTranspose);
        
        //update the weights
        this.weightsI_H.add2Matrix(deltaweightsI_H);
        
        // Adjust the Bias by its deltas (which is just the gradients)
        this.hiddenBias.add2Matrix(hidden_gradient);
        
        //code to calculate the error at each step for given digit
        double[] output_map = outputErrorsMatrix.convertMatrixToArray(outputErrorsMatrix);
           return output_map[index];
    }
    
}