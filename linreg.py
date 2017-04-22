# linreg.py
#
# Standalone Python/Spark program to perform linear regression.
# Performs linear regression by computing the summation form of the
# closed form expression for the ordinary least squares estimate of beta.
# 
# TODO: To compute the beta coeffiecients for beta values using linear regrassion and gradient descent.
# 
# Takes the yx file as input, where on each line y is the first element 
# and the remaining elements constitute the x.
#
# Usage: spark-submit linreg.py <inputdatafile>
# Example usage: spark-submit linreg.py yxlin.csv
#
#

#Jay Shah 800326050
import sys
import numpy as np

from pyspark import SparkContext

# get the xy values for th eline
def get_B(line):
    # we need y so save it
    y = float(line[0])
    # replace the first element with a float 1
    line[0] = float(1)
    # turn the line into a matrix
    line_matrix = np.asmatrix(np.array(line).astype('float')).T
    # dot product line matrix (X) with the y value and return it
    return np.dot(line_matrix,y)


def get_XXT(line):
    # we need the x's and turn them into an array and transform
    #we dont really need y so re-set that
    line[0] = float(1)
    #convert your line (which is full of numbers, to a matrix, and then make them all float
    line_matrix = np.asmatrix(np.array(line).astype('float'))
    return np.dot(line_matrix.T,line_matrix)

if __name__ == "__main__":
  if len(sys.argv) !=2:
    print >> sys.stderr, "Usage: linreg <datafile>"
    exit(-1)

  sc = SparkContext(appName="LinearRegression")

  # Input yx file has y_i as the first element of each line 
  # and the remaining elements constitute x_i
  yxinputFile = sc.textFile(sys.argv[1])
  # y0 will be the yi and the rest will be the x^T array
  yxlines = yxinputFile.map(lambda line: line.split(','))

  # we are getting the A, for calculation, this consist of a map/reduce step
  The_A_data = yxlines.map(lambda line : ("A",get_XXT(line))).reduceByKey(lambda a,b: np.add(a,b))
  A = The_A_data.map(lambda x_t :x_t[1]).collect()[0]
  #turn the The A data into an np array
  XXT_FINAL = np.asmatrix(A)

  The_B_data = yxlines.map(lambda line : ("B",get_B(line))).reduceByKey(lambda a,b:np.add(a,b))
  B = The_B_data.map(lambda x_y :x_y[1]).collect()[0]
  XY_FINAL = np.asmatrix(B)

  # take the inverse of the XXT and then dot product that witht he XY final
  beta = np.dot(np.linalg.inv(XXT_FINAL),XY_FINAL).tolist()



  # print the linear regression coefficients in desired output format
  print "beta: CLOSED FORM"
  for coeff in beta:
      print coeff



      # *************FOR GRADIENT DESCENT, EXTRA CREDIT *********************
  # Choose an Alpha, chose .01 vecause its small yet not too small
  alpha = 0.01
  #declare X and Y lists for the formula
  X = []
  Y = []
  #these are all the little x^T in the big X matrix
  x_little = []
  # i am going to use the closed form beta matrix as my initial guess.
  # if the beta that we got from closed form is truly the correct beta, then grdient descent shoulnd't
  # change the value any
  beta_matrix = np.asmatrix(np.array(beta).astype('float'))
  #get the values from the input again
  grad_lines =yxinputFile.collect()
  line_array =[]
  #iterate through each line, split it, get the y value (at the begginig of each line)
  # get the x value(s) for each y value and then append each set of x values to the X matrix
  for line in grad_lines:
      line_array = line.split(',')
      Y.append(line_array[0])
      list_x = [float(1),]
      for i in range(1,len(line_array)):
          list_x.append(line_array[i])
      x_little.append(list_x)
  X.append(x_little)
  #turn both the x and y lists to matrices, make sure each value is the a float even if they already are
  X_matrix = np.asmatrix(np.array(X).astype('float'))
  Y_matrix = np.asmatrix(np.array(Y).astype('float'))

  # similary is just a convergence value if every value in simlary matrix is less than .001 difference
  # from old beta and new beta, stop the while loop
  # make the similary matrix of ones of the same dimensions as our previous beta
  similarity = np.ones(beta_matrix.shape, dtype=np.float)
  while (np.absolute(similarity) >.001).all():
        # initial beta is the iniial beta matrix (my initial guess)
        initial_beta = beta_matrix
        # use the gradient descent formula to get a new beta value
        beta_matrix = np.add( beta_matrix, alpha * np.dot(X_matrix.T , np.subtract(Y_matrix, np.dot(X_matrix, beta_matrix))))
        # subtract old and new beta value
        similarity = np.subtract(initial_beta,beta_matrix).flatten()


    # print the linear regression coefficients in desired output format

  print "beta: GRADIENT DESCENT "
  for coeff in beta:
      print coeff

sc.stop()

