import math
import numpy as np

# FUNKCIJA F1
# @timi - corrected
class Sphere:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = 0
        self.solution = '(0,...,0)'
        self.name = "Sphere"

        for i in range(0, self.D):
            self.lb[i] = -5.12
            self.ub[i] = 5.12

    def function(self, x):
        x = np.array(x)
        result = np.sum(np.power(x, 2))
        
        return result

# @timi - corrected
# FUNKCIJA F2
class MovedAxisFunction:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = 0
        self.solution = '(0,...,0)'
        self.name = "Moved Axis Function"

        for i in range(0, self.D):
            self.lb[i] = -5.12
            self.ub[i] = 5.12

    def function(self, x):
        result = 0
        for i in range(0, len(x)):
            result += 5 * i * x[i] * x[i]
        return result


# @timi - corrected
# FUNKCIJA F8
# aka Sum function
class SumSquares:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = 0
        self.solution = '(0,...,0)'
        self.name = "Sum Squares"

        for i in range(0, self.D):
            self.lb[i] = -10
            self.ub[i] = 10

    def function(self, x):
        result = 0
        for i in range(0, len(x)):
            result += i * x[i] * x[i]

        return result

# @timi - corrected (XXXX) - name and dim not on the screenshot
# FUNKCIJA F12 proveriti granice
class InvertedCosineWaveFunction:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = -self.D+1
        self.solution = '(0,...,0)'
        self.name = 'F12'

        for i in range(0, self.D):
            self.lb[i] = -100
            self.ub[i] = 100

    def function(self, sol):
        top = 0
        for i in range(0, len(sol) - 1):
            top = top + math.exp(-(math.pow(sol[i], 2) + math.pow(sol[i + 1], 2) + 0.5 * sol[i] * sol[i + 1]) / 8) * math.cos(4 * math.sqrt(math.pow(sol[i], 2) + math.pow(sol[i + 1], 2) + 0.5 * sol[i] * sol[i + 1]))

        return -top
    
# @timi - corrected
class Pathological:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = 0
        self.solution = '(0,...,0)'
        self.name = 'Pathological'

        for i in range(0, self.D):
            self.lb[i] = -100
            self.ub[i] = 100

    def function(self, x):
        
        result = 0
        for i in range(len(x)-1):
            
            result += np.power((0.5 + ((np.power(np.sin(np.sqrt(100 * x[i]**2 + x[i+1]**2)) , 2) - 0.5) / (1 + 0.001 * (x[i]**2 - 2 * x[i] * x[i+1] + x[i+1]**2)**2 ))), 2)

        return result  

# @timi - corrected - dim not on the screenshot
# FUNKCIJA F14 proveriti granice
class Discus:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = 0
        self.solution = '(0,...,0)'
        self.name = 'Discus'

        for i in range(0, self.D):
            self.lb[i] = -100
            self.ub[i] = 100

    def function(self, sol):
        top = 0
        for i in range(1, self.D):
            top = top + math.pow(sol[i], 2)

        return math.pow(10, 6) * sol[0]**2 + top

# @timi - the implementation is good, the optimum <> 0, should check on the original paper
# FUNKCIJA F15 
class HappyCat:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = 0
        self.solution = '(-1,...,-1)'
        self.name = 'Happy Cat'

        for i in range(0, self.D):
            self.lb[i] = -2
            self.ub[i] = 2

    def function(self, sol):
        expr1 = 0
        expr2 = 0
        expr3 = 0
        for i in range(0, self.D):
            expr1 = expr1 + math.pow(sol[i], 2) - self.D
            expr2 = expr2 + math.pow(sol[i], 2)
            expr3 = expr3 + sol[i]

        return math.pow(abs(expr1), 1/8) + (0.5 * expr2 + expr3) / self.D + 0.5


# @timi - corrected
class Shekel07:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = -10.4028
        self.solution = '(4,4,4,4)'
        self.name = 'Shekel 07'

        for i in range(0, self.D):
            self.lb[i] = 0
            self.ub[i] = 10

    def function(self, x):
        C = np.array([
                    [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
                    [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 5.0, 1.0, 2.0, 3.6],
                    [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 3.0, 8.0, 6.0, 7.0],
                    [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6]])
        outer = 0
        b = 0.1 * np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5])
        m = 7
        for i in range(1, m+1):
            bi = b[i-1]
            inner = 0
            for j in range(1, 5):
                xj = x[j-1]
                Cji = C[j-1, i-1]
                inner = inner + (xj-Cji)**2
            outer = outer + 1/(inner+bi)
        
        return -outer


    
# @timi - corrected
class Shekel05:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = -10.1532
        self.solution = '(4,4,4,4)'
        self.name = 'Shekel 05'

        for i in range(0, self.D):
            self.lb[i] = 0
            self.ub[i] = 10

    def function(self, x):
        C = np.array([
                    [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
                    [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 5.0, 1.0, 2.0, 3.6],
                    [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 3.0, 8.0, 6.0, 7.0],
                    [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6]])
        outer = 0
        b = 0.1 * np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5])
        m = 5
        for i in range(1, m+1):
            bi = b[i-1]
            inner = 0
            for j in range(1, 5):
                xj = x[j-1]
                Cji = C[j-1, i-1]
                inner = inner + (xj-Cji)**2
                
            outer = outer + 1/(inner+bi)
        
        return -outer
    
# @timi - corrected
class Schwefel_2_26:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = -418.9829 * self.D
        self.solution = 420.9687
        self.name = 'Schwefel 2.26'

        for i in range(0, self.D):
            self.lb[i] = -500
            self.ub[i] = 500

    def function(self,x):
        
        x = np.array(x)
        
        result = np.sum(x * np.sin(np.sqrt(np.abs(x))))
           
        return 418.9829 * self.D - result
    
# @timi - corrected
#FUNKCIJA F9
class Schwefel_2_22:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = 0
        self.solution = '(0,...,0)'
        self.name = 'Schwefel 2.22'

        for i in range(0, self.D):
            self.lb[i] = -10
            self.ub[i] = 10

    def function(self, x):
        
        x = np.array(x)
        result = np.sum(np.abs(x))+np.prod(np.abs(x));
        
        return result
  #  $f(x^{*})=0$ at $x^{*}=(0,...,0)$ 

# @timi - corrected
class Schwefel_2_21:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = 0
        self.solution = '(0,...,0)'
        self.name = 'Schwefel 2.21'

        for i in range(0, self.D):
            self.lb[i] = -100
            self.ub[i] = 100

    def function(self, x):
        x = np.array(x)
        return np.max(np.abs(x))

# @timi - corrected
class Schwefel_1_2:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = 0
        self.solution = '(0,...,0)'
        self.name = 'Schwefel 1.2'

        for i in range(0, self.D):
            self.lb[i] = -100
            self.ub[i] = 100

    def function(self, x):
        x = np.array(x)
        return np.sum([np.sum(x[:i]) ** 2 for i in range(len(x))])

#FUNKCIJA F17
class Schaffer2:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = 0
        self.solution = '(0,...,0)'
        self.name = 'Schaffer 2'

        for i in range(0, self.D):
            self.lb[i] = -100
            self.ub[i] = 100

    def function(self, sol):
        x1 = sol[0]
        x2 = sol[1]

        return 0.5 + (math.pow(math.sin((math.pow(x1, 2) - math.pow(x2, 2))), 2) - 0.5) / math.pow((1 + 0.001 * (math.pow(x1, 2) + math.pow(x2, 2))), 2)

class Schaffer1:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.name = 'Schaffer 1'

        for i in range(0, self.D):
            self.lb[i] = -100
            self.ub[i] = 100

    def function(self, sol):
        x1 = sol[0]
        x2 = sol[1]

        return 0.5 + (math.pow(math.sin(math.pow((math.pow(x1, 2) + math.pow(x2, 2)), 2)), 2) - 0.5)/ math.pow((1 + 0.001 * (math.pow(x1, 2) + math.pow(x2, 2))), 2)

class Salomon:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = 0
        self.solution = '(0,...,0)'
        self.name = 'Salomon'

        for i in range(0, self.D):
            self.lb[i] = -100
            self.ub[i] = 100

    def function(self, sol):
        sum = 0
        
        for i in range(len(sol)):
            sum += math.pow(sol[i], 2)
        
        return 1 - math.cos(2 * math.pi * math.sqrt(sum)) + 0.1 * math.sqrt(sum)

# @timi - corrected
class Zakharov:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = 0
        self.solution = '(0,...,0)'
        self.name = 'Zakharov'

        for i in range(0, self.D):
            self.lb[i] = -5
            self.ub[i] = 10

    def function(self, sol):
        sum1 = 0
        sum2 = 0

        for i in range(len(sol)):
            sum1 += math.pow(sol[i],2)
            sum2 += 0.5*i*sol[i]
        
        
        return sum1 + math.pow(sum2,2) + math.pow(sum2,4)

# @timi - corrected
class Step02:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = 0
        self.solution = '(0,...,0)'
        self.name = 'Step 2'

        for i in range(0, self.D):
            self.lb[i] = -100
            self.ub[i] = 100

    def function(self, x):
        x = np.array(x)
        result = np.sum(np.abs((x + 0.5))**2);
        return result

# @timi - corrected
class Shekel10:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = -10.5363
        self.solution = '(4,4,4,4)'
        self.name = 'Shekel 10'

        for i in range(0, self.D):
            self.lb[i] = 0
            self.ub[i] = 10

    def function(self, x):
        C = [[4,4,4,4],
             [1,1,1,1],
             [8,8,8,8],
             [6,6,6,6],
             [3,7,3,7],
             [2,9,2,9],
             [5,5,3,3],
             [8,1,8,1],
             [6,2,6,2],
             [7,3.6,7,3.6]]
        b = [.1,.2,.2,.4,.4,.6,.3,.7,.5,.5]
        C = np.asarray(C)
        b = np.asarray(b)
        result = 0;
        for i in range(0,9):
          v = np.matrix(x-C[i,:])
          result = result - ((v)*(v.T)+b[i])**(-1)
        
        return result.item(0)  
    # $f(x^{*})=-10.5363$ at $x^{*}=(4,4,4,4)$ 

# @timi - corrected 
class Rosenbrock:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = 0
        self.solution = '(1,...,1)'
        self.name = 'Rosenbrock'

        for i in range(0, self.D):
            self.lb[i] = -10
            self.ub[i] = 10

    def function(self, x):
        x = np.array(x)
        result = 0
        for i in range(len(x)-1):
            
            result += (100 * (x[i + 1] - x[i] ** 2) ** 2) + ((x[i] - 1) ** 2) 
            

       # result = np.sum([((x[i] - 1) ** 2) + (100 * (x[i + 1] - x[i] ** 2) ** 2) for i in range(len(x) - 1)])
       
        return result

# @timi - corrected 
class Rastrigin:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = 0
        self.solution = '(0,...,0)'
        self.name = 'Rastrigin'

        for i in range(0, self.D):
            self.lb[i] = -5.12
            self.ub[i] = 5.12

    def function(self, x):
        x = np.array(x)
        result = 10 * len(x) + (x ** 2 - 10 * np.cos(2 * np.pi * x)).sum()

        return result


# @timi - corrected
# Quartic with noise
class Quartic:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = 0
        self.solution = '(0,...,0)'
        self.name = 'Quartic'

        for i in range(0, self.D):
            self.lb[i] = -1.28
            self.ub[i] = 1.28

    def function(self, x):
        x = np.array(x)
        result = 0
        for i in range(len(x)):
            result += i * np.power(x[i], 4) 
        
        return result + np.random.random()
   # $f(x^{*})=0$ at $x^{*}=(0,...,0)$ 
   
# @timi - corrected
# FUNKCIJA F7
class PowellSum:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = 0
        self.solution = '(0,...,0)'
        self.name = 'Powell Sum'

        for i in range(0, self.D):
            self.lb[i] = -100
            self.ub[i] = 100

    def function(self, x):
        result = 0
        for i in range(0, len(x)):
            result += np.power(np.abs(x[i]), i + 1)

        return result


class PowellsQuadratic:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.name = 'Powells Quadratic'

        for i in range(0, self.D):
            self.lb[i] = -10
            self.ub[i] = 10

    def function(self, sol):
        x1 = sol[0]
        x2 = sol[1]
        x3 = sol[2]
        x4 = sol[3]

        return math.pow((x1 + 10 * x2), 2) + 5 * math.pow((x3 - x4), 2) + math.pow((x2 - 2 * x3), 4) + 10 * math.pow((x1 - x4), 4) 

# @timi - corrected
#FUNKCIJA F10
class PowellSingular:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = 0
        self.solution = '(0,...,0)'
        self.name = 'Powell Singular'

        for i in range(0, self.D):
            self.lb[i] = -4
            self.ub[i] = 5

    def function(self, x):
        dim = len(x)
        term1 = 0
        term2 = 0
        term3 = 0
        term4 = 0

        for i in range(int(dim / 4)):
            term1 += np.power(x[4 * i - 3] + 10 * x[4 * i - 2], 2)
            term2 += 5 * np.power(x[4 * i - 1] - x[4 * i], 2)
            term3 += np.power(x[4 * i - 2] - 2 * x[4 * i - 1], 4)
            term4 += 10 * np.power(x[4 * i - 3] - x[4 * i], 4)

        return term1 + term2 + term3 + term4

# @timi - corrected
class Perm:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = 0
        self.solution = '(1, 2, 3, D)'
        self.name = 'Perm'

        for i in range(0, self.D):
            self.lb[i] = -10
            self.ub[i] = 10

    def function(self, x):
        tmp1 = 0
        tmp2 = 0
        beta = 1
        for j in range(len(x)):
            for i in range(len(x)):
                tmp1 += (i+1+beta)*(np.power(x[i],j+1)-np.power(1/(i+1),j+1))
            tmp2 += np.power(tmp1,2)
            tmp1 = 0
        return tmp2

# @timi - corrected
class f12:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = 0
        self.solution = '(1,...,1)'
        self.name = 'Penalty 1'

        for i in range(0, self.D):
            self.lb[i] = -50
            self.ub[i] = 50

    def function(self, x):
        
        x = np.array(x)
        dim = len(x)
        result = (np.pi / dim) * (10 * ((np.sin(math.pi*(1+(x[0]+1)/4)))**2) + np.sum((((x[1:dim-1]+1)/4)**2)*(1+10*((np.sin(np.pi*(1+(x[1:dim-1]+1)/4))))**2))+((x[dim-1]+1)/4)**2) + np.sum(self.Ufun(x,10,100,4))
   
        return result
    
    def Ufun(self, x,a,k,m): 
        y=k*((x-a)**m)*(x>a)+k*((-x-a)**m)*(x<(-a));
        return y

# @timi - corrected
class Penalty2:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = 0
        self.solution = '(1,...,1)'
        self.name = 'Penalty 2'

        for i in range(0, self.D):
            self.lb[i] = -50
            self.ub[i] = 50

    def function(self, x):
        
        x = np.array(x)
        dim = len(x)
        result = 0.1 * ((np.sin(3 * np.pi * x[1]))**2 + np.sum((x[0:dim-2]-1)**2 * (1 + (np.sin(3 * np.pi * x[1:dim-1]))**2)) + ((x[dim-1]-1)**2) * (1 + (np.sin(2 * np.pi * x[dim-1]))**2)) + np.sum(self.Ufun(x,5,100,4)) 
        return result
    
    def Ufun(self, x,a,k,m): 
        y=k*((x-a)**m)*(x>a)+k*((-x-a)**m)*(x<(-a));
        return y

class MultiGaussian:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = 0
        self.solution = '(0,...,0)'
        self.name = 'Multi Gaussian'

        for i in range(0, self.D):
            self.lb[i] = -2
            self.ub[i] = 2

    def function(self, sol):
        x1 = sol[0]
        x2 = sol[1]

        a = [0.5, 1.2, 1.0, 1.0, 1.2]
        b = [0.0, 1.0, 0.0, -0.5, 0.0]
        c = [0.0, 0.0, -0.5, 0.0, 1.0]
        d = [0.1, 0.5, 0.5, 0.5, 0.5]

        sum = 0.0


        for i in range(0, 5):
            sum += a[i] * math.exp(-(math.pow(x1 - b[i], 2) + math.pow(x2 - c[i], 2)) / math.pow(d[i], 2))
        
        return sum

class ModifiedRosenbrock:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.name = 'Modified Rosenbrock'

        for i in range(0, self.D):
            self.lb[i] = -5
            self.ub[i] = 5

    def function(self, sol):
        x1 = sol[0]
        x2 = sol[1]

        return 100 * math.pow((x2 - math.pow(x1, 2)), 2) + math.pow((6.4 * math.pow((x2 - 0.5), 2) - x1 - 0.6), 2)


class MieleAndCantrell:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.name = 'Miele and Cantrell'

        for i in range(0, self.D):
            self.lb[i] = -1
            self.ub[i] = 1

    def function(self, sol):
        x1 = sol[0]
        x2 = sol[1]
        x3 = sol[2]
        x4 = sol[3]
        
        return math.pow((math.exp(-x1) - x2), 4) + 100 * math.pow((x2 - x3), 6) + math.pow((math.tan(x3 - x4)), 4) + math.pow(x1, 8)

class MeyerAndRoth:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.name = 'Meyer and Roth'

        for i in range(0, self.D):
            self.lb[i] = -10
            self.ub[i] = 10

    def function(self, sol):
        x1 = sol[0]
        x2 = sol[1]
        x3 = sol[2]

        t = [1, 2, 1, 2, 0.1]
        v = [1, 1, 2, 2, 0]
        y = [0.126, 0.219, 0.076, 0.126, 0.186]

        sum = 0

        for i in range(0, 6):
            sum += math.pow(((x1 * x3 * t[i]) / (1 + x1 * t[i] + x2 * v[i]) - y[i]), 2)
        
        return sum

class LevyAndMontalvo2:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.name = 'Levy and Montalvo 2'

        for i in range(0, self.D):
            self.lb[i] = -5
            self.ub[i] = 5

    def function(self, sol):
        n = len(sol)
        exp2 = 0

        x1 = sol[0]
        xn = sol[n - 1]

        exp1 = math.pow(math.sin(3 * math.pi * x1), 2)

        for i in range(0, n-1):
            exp2 += math.pow((sol[i] - 1), 2) * (1 + math.pow(math.sin(3 * math.pi * sol[i] + 1), 2))
        
        
        exp3 = math.pow((xn - 1), 2) * (1 + math.pow((math.sin(2 * math.pi * xn)), 2))
        return 0.1 * (exp1 + exp2) + exp3

# @timi - corrected
# Levy function
class LevyAndMontalvo1:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = 0
        self.solution = '(1,...,1)'
        self.name = 'Levy and Montalvo 2'

        for i in range(0, self.D):
            self.lb[i] = -5.12
            self.ub[i] = 5.12

    def function(self, x):
        x = np.array(x)
        z = 1 + (x - 1) / 4
        result = (np.sin(np.pi * z[0] )**2 + np.sum((z[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * z[:-1] + 1)**2 )) + (z[-1] - 1)**2 * (1 + np.sin(2 * np.pi * z[-1] )**2 ))
        return result
# @timi - corrected
class Kowalik:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = 0.00030748610
        self.solution = '(0.192833, 0.190836, 0.123117, 0.135766)'
        self.name = 'Kowalik'

        for i in range(0, self.D):
            self.lb[i] = -5
            self.ub[i] = 5

    def function(self, x):
        x = np.array(x)

        a = [0.1957, 0.1947, 0.1735, 0.1600, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246]
        b = [4, 2, 1, 1 / 2, 1 / 4, 1 / 6, 1 / 8, 1 / 10, 1 / 12, 1 / 14, 1 / 16]

        a = np.array(a)
        b = np.array(b)
        
        result = np.sum((a- ((x[0] * (b**2 + x[1] * b)) / (b**2 + x[2] * b + x[3])))**2)

        return result
    # $f(x^{*})=0.00030748610$ at $x^{*}= [0.192833, 0.190836, 0.123117, 0.135766]$ 


# @timi - double check
class Hosaki:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = -2.3458
        self.solution = '(3,2)'
        self.name = 'Hosaki'

        for i in range(0, self.D):
            self.lb[i] = 0
            self.ub[i] = 10

    def function(self, sol):
        x1 = sol[0]
        x2 = sol[1]

        return (1 - 8 * x1 + 7 * x1**2 - 7 / 3 * x1**3 + 1 / 4 * x1**4) * (x2**2 * np.exp(-x2))
    

# @timi - corrected
class Holzman2:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = 0
        self.solution = '(0,...,0)'
        self.name = 'Holzman 2'

        for i in range(0, self.D):
            self.lb[i] = -10
            self.ub[i] = 10

    def function(self, sol):
        result = 0
        for i in range(len(sol)):
            result += i * np.power(sol[i],4)

        return result

# @Timi - corrected
class Hartman06:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = -3.32236801141551
        self.solution = '~(0.20168952, 0.15001069, 0.47687398, 0.27533243, 0.31165162, 0.65730054)'
        self.name = 'Hartman 06'
        
        for i in range(0, self.D):
            self.lb[i] = 0
            self.ub[i] = 1

    def function(self,x):

        a = np.empty([4, 6])
        a[0,0]=10.0;	a[0,1]=3.0;		a[0,2]=17.0;	a[0,3]=3.5;		a[0,4]=1.7;		a[0,5]=8.0
        a[1,0]=0.05;	a[1,1]=10.0;	a[1,2]=17.0;	a[1,3]=0.1;		a[1,4]=8.0;		a[1,5]=14.0
        a[2,0]=3.0;		a[2,1]=3.5;		a[2,2]=1.7;		a[2,3]=10.0;	a[2,4]=17.0;	a[2,5]=8.0
        a[3,0]=17.0;	a[3,1]=8.0;		a[3,2]=0.05;	a[3,3]=10.0;	a[3,4]=0.1;		a[3,5]=14.0

        c = np.empty([4])
        c[0]=1.0;   c[1]=1.2;   c[2]=3.0;   c[3]=3.2

        p = np.empty([4, 6])
        p[0,0]=0.1312;	p[0,1]=0.1696;	p[0,2]=0.5569;	p[0,3]=0.0124;	p[0,4]=0.8283;	p[0,5]=0.5886
        p[1,0]=0.2329;	p[1,1]=0.4135;	p[1,2]=0.8307;	p[1,3]=0.3736;	p[1,4]=0.1004;	p[1,5]=0.9991
        p[2,0]=0.2348;	p[2,1]=0.1451;	p[2,2]=0.3522;	p[2,3]=0.2883;	p[2,4]=0.3047;	p[2,5]=0.6650
        p[3,0]=0.4047;	p[3,1]=0.8828;	p[3,2]=0.8732;	p[3,3]=0.5743;	p[3,4]=0.1091;	p[3,5]=0.0381

        s = 0

        for i in range(1, 5):
            sm = 0
            for j in range(1, 7):
                
                sm = sm+a[i-1,j-1]*(x[j-1]-p[i-1,j-1])**2
            s = s+c[i-1]*np.exp(-sm)

        y = -s
        return y    
    
    
# @Timi - corrected
class Hartman03:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = -3.86278214782076
        self.solution = '~(0.1,0.55592003, 0.85218259)'

        for i in range(0, self.D):
            self.lb[i] = 0
            self.ub[i] = 1
    
    def function(self, x):
        alpha = [1.0, 1.2, 3.0, 3.2]
        A = np.array([
                    [3.0, 10, 30],
                    [0.1, 10, 35],
                    [3.0, 10, 30],
                    [0.1, 10, 35]])
        P = 10.0**(-4.0) * np.array([
                    [3689, 1170, 2673],
                    [4699, 4387, 7470],
                    [1091, 8732, 5547],
                    [381, 5743, 8828]])
            
        outer = 0.0
        for i in range(4):
            inner = 0
            for j in range(3):
                xj = x[j]
                Aij = A[i, j]
                Pij = P[i, j]
                inner = inner + Aij*(xj-Pij)**2
            new = alpha[i] * np.exp(-inner)
            outer = outer + new
            result = - outer
        return result
# $f(x^{*})=-3.86278214782076$ at $x^{*} \cong (0.1,0.55592003,0.85218259)$ 

# @timi - double check
class GulfResearch:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = 0
        self.solution = '(50,25,1.5)'
        self.name = 'Gulf Research'

        for i in range(0, self.D):
            self.lb[i] = 0.1
            self.ub[i] = 100

    def function(self, sol):
        x1 = sol[0]
        x2 = sol[1]
        x3 = sol[2]

        m = 99
        result = 0

        for j in range(0, m):
            t = j / 100
            y = 25 + np.power((-50 * np.log(t)), 1/1.5)
            exponent = -((y - x2)**x3) / x1
            result += (np.exp(exponent) - t)**2
        
        return result

# @timi - corrected
# FUNKCIJA F3
class Griewank:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = 0
        self.solution = '(0,...,0)'
        self.name = 'Griewank'

        for i in range(0, self.D):
            self.lb[i] = -600
            self.ub[i] = 600

    def function(self, x):
        
        x = np.array(x)
        dim = len(x)
        w = [i for i in range(dim)]
        w = [i+1 for i in w]
        
        result = np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(w))) + 1
        
        return result
 
# @timi - corrected
class GoldsteinAndPrice:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = 3
        self.solution = '(0,-1)'
        self.name = 'Goldstein and Price'

        for i in range(0, self.D):
            self.lb[i] = -5
            self.ub[i] = 5

    def function(self, x):
        x1 = x[0]
        x2 = x[1]
        exp11 = (x1 + x2 + 1)**2
        exp12 = 19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2
        exp1 = 1 + exp11 * exp12
        
        exp21 = (2*x1 - 3*x2)**2
        exp22 = 18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2
        exp2 = 30 + exp21*exp22
        
        y = exp1*exp2
        return y
    # $f(x^{*})=3$ at $x^{*}=(0,-1)$ 


# @timi - corrected
class Exponential:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = -1
        self.solution = '(0,...,0)'
        self.name = 'Exponential'

        for i in range(0, self.D):
            self.lb[i] = -2
            self.ub[i] = 2
    
    def function(self, x):
        x = np.array(x)
        result = -np.exp(-0.5 * (np.sum(x**2)))
        
        return result

# @timi - corrected
# Dimension 5
# Dimension 10, fx = -9.660152 x* (2.693, 0.259, 2.074, 1.023, 2.275, 0.500, 2.138, 0.794, 2.219, 0.533)
class EpistaticMichalewicz:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = -4.687658
        self.solution = '(2.693, 0.259, 2.074, 1.023, 1.720)'
        self.name = 'Epistatic Michalewicz'

        for i in range(0, self.D):
            self.lb[i] = 0
            self.ub[i] = math.pi

    def function(self, x):
        x = np.array(x)
        m = 10
        n = len(x)
        result = -np.sum([np.sin(x[i])*np.sin(((i+1)*x[i]**2)/np.pi)**(2*m) for i in range(0, n)], axis=0)

        return result

# @timi - corrected
class Easom:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = -1
        self.solution = '(pi, pi)'
        self.name = 'Easom'

        for i in range(0, self.D):
            self.lb[i] = -100
            self.ub[i] = 100

    def function(self, sol):
        x1 = sol[0]
        x2 = sol[1]

        return -math.cos(x1) * math.cos(x2) * math.exp(-math.pow(x1 - math.pi, 2) - math.pow(x2 - math.pi, 2))

# @timi - corrected
class DixonAndPrice:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = 0
        self.solution = '2**-(2**i/2**i)'
        self.name = 'Dixon and Price'

        for i in range(0, self.D):
            self.lb[i] = -10
            self.ub[i] = 10

    def function(self, x):
        x = np.array(x)
        result = (x[0] - 1)**2 + np.sum([(i+1)*(2*x[i]**2 - x[i-1])**2 for i in range(1, 2)], axis=0)
        return result

# @timi - corrected 
# Dimension 2
class DekkersAndAarts:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = -24777
        self.solution = '(0,15) and (0,-15)'
        self.solution = 'Dekkers and Aarts'
        
        for i in range(0, self.D):
            self.lb[i] = -20
            self.ub[i] = 20

    def function(self, sol):
        x1 = sol[0]
        x2 = sol[1]

        return math.pow(10, 5) * math.pow(x1, 2) - math.pow((math.pow(x1, 2) + math.pow(x2, 2)), 2) + math.pow(10, -5) * math.pow((math.pow(x1, 2) + math.pow(x2, 2)), 4)

# @timi - corrected 
# (Discontinuous, Non-Differentiable, Separable, Scalable, Multimodal)
# Dimension = 2,4
class CosineMixture:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = -self.D*0.1
        self.solution = '(0,0)'
        self.name = 'Cosine Mixture'
        
        for i in range(0, self.D):
            self.lb[i] = -1
            self.ub[i] = 1
            
    
    def function(self, x):
        x = np.array(x)
        result =  np.sum(x**2) - 0.1 * np.sum(np.cos(5*np.pi*x)) 
        return result

#FUNKCIJA F18
# @timi - corrected
class CamelBackThreeHump:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = 0
        self.solution = '(0,...,0)'
        self.name = 'Camel Back Three Hump'
        
        for i in range(0, self.D):
            self.lb[i] = -5
            self.ub[i] = 5

    def function(self, sol):
        x1 = sol[0]
        x2 = sol[1]

        return 2 * math.pow(x1, 2) - 1.05 * math.pow(x1, 4) + math.pow(x1, 6) / 6 + x1 * x2 + math.pow(x2, 2)

# @timi - corrected
class CamelBackSixHump:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = -1.031628453489877
        self.solution = '~(+-0.08984201368301331,+-0.7126564032704135)'
        self.name = 'Camel Back Six Hump'

        for i in range(0, self.D):
            self.lb[i] = -5
            self.ub[i] = 5

    def function(self, x):
        x = np.array(x)
        result = 4 * (x[0]**2) - 2.1 * (x[0]**4) + (x[0]**6) / 3 + x[0] * x[1] - 4 * (x[1]**2) + 4 * (x[1]**4)
        return result
# $f(x^{*})=-1.031628453489877$ at $x^{*}= \approx (\pm0.08984201368301331,\pm0.7126564032704135)$ 


# @timi - corrected
class Brown:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = 0
        self.solution = '(0,...,0)'
        self.name = 'Brown'
        
        for i in range(0, self.D):
            self.lb[i] = -1
            self.ub[i] = 4

    def function(self, sol):
        result = 0
        for i in range(len(sol)-1):
            result += np.power(np.power(sol[i], 2), (np.power(sol[i + 1],2) + 1)) + np.power(np.power(sol[i+1],2), (np.power(sol[i], 2) + 1))
        
        return result

# @timi - corrected
class Branin:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = 0.39788735772973816
        self.solution = '~(+- 0.08984201368301331,+- 0.7126564032704135)'
        self.name = 'Branin'

        for i in range(0, self.D):
            self.lb[i] = -5
            self.ub[i] = 15

    def function(self, x):
        x = np.array(x)
        result = (x[1] - (x[0]**2) * 5.1 / (4 * (np.pi**2)) + 5 /np.pi * x[0] - 6)**2 + 10 * (1 - 1 / (8 *np.pi)) * np.cos(x[0]) + 10
        return result
# $f(x^{*})=0.39788735772973816$ at $x^{*}= \approx (\pm0.08984201368301331,\pm0.7126564032704135)$ 
        
# @timi - corrected
class Bohachevsky1:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = 0
        self.solution = '(0,...,0)'
        self.name = 'Bohachevsky 1'

        for i in range(0, self.D):
            self.lb[i] = -100
            self.ub[i] = 100

    def function(self, sol):
        x1 = sol[0]
        x2 = sol[1]

        return math.pow(x1, 2) + 2 * math.pow(x2, 2) - 0.3 * math.cos(3 * math.pi * x1) - 0.4 * math.cos(4 * math.pi * x2) + 0.7

# @timi - corrected
class Bohachevsky2:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = 0
        self.solution = '(0,...,0)'
        self.name = 'Bohachevsky 2'

        for i in range(0, self.D):
            self.lb[i] = -100
            self.ub[i] = 100

    def function(self, sol):
        x1 = sol[0]
        x2 = sol[1]

        return math.pow(x1, 2) + 2 * math.pow(x2, 2) - 0.3 * math.cos(3 * math.pi * x1) * math.cos(4 * math.pi * x2) + 0.3


# @timi - corrected
class BeckerAndLago:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = 0
        self.solution = '(pm5,..., pm5)'
        self.name = 'Becker and Lago'

        for i in range(0, self.D):
            self.lb[i] = -100
            self.ub[i] = 100

    def function(self, sol):
        x1 = sol[0]
        x2 = sol[1]

        return math.pow((abs(x1) - 5), 2) + math.pow((abs(x2) - 5), 2)

# @timi - corrected
# Zirilli or Aluffi-Pentini’s Function (Continuous, Differentiable, Separable, Non-Scalable, Unimodal)
class AluffiPentini:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = -0.3523
        self.solution = '(−1.0465, 0)'
        self.name = 'Aluffi Pentini'
        

        for i in range(0, self.D):
            self.lb[i] = -10
            self.ub[i] = 10
    
    def function(self, sol):
        sol = np.array(sol)
        x1 = sol[0]
        x2 = sol[1]

        return 0.25 * math.pow(x1, 4) - 0.5 * math.pow(x1, 2) + 0.1 * x1 + 0.5 * math.pow(x2, 2)

# @timi - corrected
#FUNKCIJA F11
class Alpine:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = 0
        self.solution = '(0,...,0)'
        self.name = 'Alpinie'

        for i in range(0, self.D):
            self.lb[i] = -10
            self.ub[i] = 10
    
    def function(self, x):
        x = np.array(x)
        result = np.sum(np.abs(x * np.sin(x + 0.1 * x)))
        
        return result

# @timi - corrected
#Funkcija F6
class Ackley:

    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = 0
        self.solution = '(0,...,0)'
        self.name = 'Ackley'

        for i in range(0, self.D):
            self.lb[i] = -32
            self.ub[i] = 32

    def function(self, x):

        x = np.array(x)
        dim = len(x)
        result = -20 * np.exp(-0.2*np.sqrt(np.sum(x**2)/dim)) - np.exp(np.sum(np.cos(2*np.pi*x))/dim) + 20 + np.exp(1)

        return result
    
# @timi - corrected
#Funkcija F16
class DropWaveFunction:

    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = -1
        self.solution = '(0,...,0)'
        self.name = 'Drop Wave Function'

        for i in range(0, self.D):
            self.lb[i] = -5.2
            self.ub[i] = 5.2

    def function(self, x):
        
        x = np.array(x)
        
        x1 = x[0]
        x2 = x[1]

        return (1 + math.cos(12 * math.sqrt(math.pow(x1, 2) + math.pow(x2, 2)))) / (0.5 * (math.pow(x1, 2) + math.pow(x2, 2)) + 2)


# https://www.sfu.ca/~ssurjano/rothyp.html#:~:text=The%20Rotated%20Hyper%2DEllipsoid%20function,shows%20its%20two%2Ddimensional%20form.
class RotatedHyperEllipsoid:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = 0
        self.solution = '(0,...,0)'
        self.name = "Rotated Hyper-Ellipsoid"

        for i in range(0, self.D):
            self.lb[i] = -65
            self.ub[i] = 65

    def function(self, x):
        outer = 0
        for i in range(0, len(x)):
            inner = 0
            for j in range(i):
                x_j = x[j]
                inner = inner + x_j**2

            outer = outer + inner
        return outer
    
#https://www.sfu.ca/~ssurjano/trid.html
class Trid:
    def __init__(self, D):
        self.D = D
        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = -self.D * (self.D + 4) * (self.D - 1)/6
        self.solution = 'x_i = i(d+1-i)'
        self.name = "Trid"

        for i in range(0, self.D):
            self.lb[i] = -self.D
            self.ub[i] = self.D

    def function(self, x):
        d = len(x)
        sum1 = (x[0]-1)**2
        sum2 = 0

        for i in range(1, d):
            x_i = x[i]
            x_old = x[i-1]
            sum1 = sum1 + (x_i-1)**2
            sum2 = sum2 + x_i * x_old

        return sum1 - sum2
    
    


    
    
    
    