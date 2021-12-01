In order to run the program: 

type in the following command : 

spark-submit linreg.py yxlin2.csv > /users/jdshah/Answers/yxlin2.out

* linreg.py is your python file for your code
* yxlin2.csv is your input file
* >/users/jdshah/Answers/yxlin2.txt is your output file path, anything you
  print in the progrma will be written to a file, in my case it is the 
  closed form beta values and the gradient descent beta values


Gradient Descent: 

* I used a while loop to see if the old beta and new beta difference is less
  than .001. once the values were within that difference range. The while loop
  exits and then we print the converged beta values. I used a matrix 
  of ones of the same dimension as the beta values as I got from the 
  closed form calculation of beta matrix. Even though I used Ones as a starting 
  point, I still got the same converged values as the closed form. 


* My results for both yxlin and yxlin2 are in the Answers folder provided
