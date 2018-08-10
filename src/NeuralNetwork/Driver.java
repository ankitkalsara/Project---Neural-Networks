package NeuralNetwork;

import java.awt.Color;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

public class Driver {
    
    public static void main(String[] args) throws FileNotFoundException, IOException{
        
        ArrayList<double[]> target_list = new ArrayList<double[]>();
        ArrayList<double[]> input_list = new ArrayList<double[]>();
       
        String workingDir = System.getProperty("user.dir");
        System.out.println("The running dataset is --> " + workingDir + "\\train.csv");
        System.out.println();
        String csvFile = workingDir + "\\train.csv";

        Scanner scanner = null;       
        String InputLine = "";        

        try {
            scanner = new Scanner(new BufferedReader(new FileReader(csvFile)));
            scanner.hasNextLine();
            String ar = scanner.nextLine();
            //start reading the file
            while (scanner.hasNextLine()) { 
                InputLine = scanner.nextLine();
                String[] InArray = InputLine.split(",");
                double[] input_array = new double[784];
                double[] target_array = new double[10];                
                //loop through entire record
                for (int i = 0; i < InArray.length; i++) {
                    //for the first column
                    if (i == 0) {   
                        int b = Integer.parseInt(InArray[i]);   //lable
                        for (int j = 0; j < 10; j++) {  //set of classification - upto 10
                            if (j == b) {   
                                target_array[j] = 1; 
                            } else {
                                target_array[j] = 0;
                            }
                        }
                        target_list.add(target_array);  //added in target array
                    } else {    //put the remaining elements in the input array
                        input_array[i - 1] = Double.parseDouble(InArray[i]);
                    }
                }
                input_list.add(input_array); //added in input array
            }

        } catch (Exception e) {
            System.out.println("Error occured " + e);
        }        
        
        Scanner sc = new Scanner(System.in);
        System.out.println("Enter an integer for which error graph needs to be drawn");
        int error_digit = sc.nextInt();
        System.out.println();
        
        NeuralNetwork nn = new NeuralNetwork(784, 392, 10);
        
        ArrayList<Double> avgErrorList = new ArrayList<Double>();
        
        // Training the network
        for(int x = 0; x < 20; x++)    
        {
            ArrayList<Double> error_list = new ArrayList<Double>();    
            
            for(int y = 0; y < input_list.size(); y++){ //read entire file
            
                double[] ar = target_list.get(y);
                if(ar[error_digit] == 1){
                    double a = nn.train(input_list.get(y), target_list.get(y), error_digit);
                    error_list.add(a);
                }
                //pass input_array & target_array
                nn.train(input_list.get(y), target_list.get(y), error_digit); //training individual record               
            }
            
            double sum = 0;
            
            for(int z = 0; z < error_list.size(); z++)
                sum += error_list.get(z);
            
            double avg = sum/error_list.size();
            avgErrorList.add(avg);
            System.out.println("Training# " + x + " completed");
        }        
        System.out.println();
        System.out.println("Error list for input digit = " + avgErrorList);        
        System.out.println();            
        
        
        //saving the graph
        Plot plot = Plot.plot(Plot.plotOpts().title("Graph of error generated during training of NN for given digit (multiplied by 100)").legend(Plot.LegendFormat.BOTTOM)).
			xAxis("x", Plot.axisOpts().range(0, 10)).
			yAxis("y", Plot.axisOpts().range(0, 100)).
			series("Errors", Plot.data().xy(1, avgErrorList.get(0)*100).
                                                     xy(2, avgErrorList.get(1)*100).
                                                     xy(3, avgErrorList.get(2)*100).
                                                     xy(4, avgErrorList.get(3)*100).
                                                     xy(5, avgErrorList.get(4)*100).
                                                     xy(6, avgErrorList.get(5)*100).
                                                     xy(7, avgErrorList.get(6)*100).
                                                     xy(8, avgErrorList.get(7)*100).
                                                     xy(9, avgErrorList.get(8)*100).
                                                     xy(10, avgErrorList.get(9)*100),Plot.seriesOpts().
					marker(Plot.Marker.DIAMOND).
					markerColor(Color.CYAN).
					color(Color.BLACK));
		plot.save("Graph for Arabic digits set", "png");
                System.out.println("Graph saved !");
                System.out.println();
        
        System.out.println("Neural network has been trained now !");        
        System.out.println();
        
        //start predicting the output           
        int b;        
        for(int i = 0; i < 15; i++){  //provide the range for which prediction needs to be done
            b = i + 2;  //for keeping track of excel row
            double[] arr = target_list.get(i); //target data            
            
            System.out.println("Target dataset of "  +  b + "th" + " record = " + Arrays.toString(arr));
            for(int k = 0; k < arr.length; k++){                          
                if(arr[k] == 1){                    
                    System.out.println("Expected outcome from dataset = " + k);                    
                    break;
                }
            }           
            
            double[] predicted_outcome = nn.feedForward(input_list.get(i));                                
            double max = Integer.MIN_VALUE; //-2.147483648E9
            int index = 0;
            //find the maximum of predicted outcome and that will be the actual prediction !
            for(int l = 0; l < predicted_outcome.length; l++){
                if(predicted_outcome[l] > max){
                    max = predicted_outcome[l];
                    index = l;
                }
            }
            System.out.println("Predicted outcome by code = " + index);
            System.out.println();            
        }   
    }
    
}    