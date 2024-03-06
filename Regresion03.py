# Regresión Lineal (Incluye Coeficiente de Correlación y Determinación)
    
class MatematicasDiscretas():
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.N = len(X)
        self.SumatoriaX = sum(X)
        self.SumatoriaY = sum(Y)
        self.mediaY = self.SumatoriaY/self.N
    def ObtenerDatosCalcularRegresionLineal(self):
        SumatoriaXCuadrado = sum(x ** 2 for x in self.X)
        SumatoriaXY = sum(x * y for x, y in zip(self.X, self.Y))
        return self.X, self.N, self.SumatoriaX, self.SumatoriaY, SumatoriaXCuadrado, SumatoriaXY
    def ObtenerDatosCalcularCoeficienteCorrelacion(self):
        mediaX = self.SumatoriaX/self.N
        SS_XY = sum((x - mediaX) * (y - self.mediaY) for x, y in zip(self.X, self.Y))
        SS_XX = sum((x - mediaX) ** 2 for x in self.X)
        SS_YY = sum((y - self.mediaY) ** 2 for y in self.Y)
        return SS_XY, SS_XX, SS_YY
    def ObtenerDatosCalcularCoeficienteDeterminacion(self):
        return self.Y, self.mediaY

class RegresionLineal():
    def __init__(self, MatDiscretas):
        self.MatDiscretas = MatDiscretas
        self.X, self.N, self.SumX, self.SumY, self.SumXCuad, self.SumXY = MatDiscretas.ObtenerDatosCalcularRegresionLineal()
        # self.ValorX = self.X[0]
    def CalcularCoeficientes(self):
        B0 = (self.SumY * self.SumXCuad - self.SumX * self.SumXY) / (self.N * self.SumXCuad - self.SumX ** 2) # Constante
        B1 = (self.N * self.SumXY - self.SumX * self.SumY) / (self.N * self.SumXCuad - self.SumX ** 2) # Coeficiente
        return B0, B1
    def predecir_y(self, X_prediccion):
        # self.ValorX = self.X[0]
        B0, B1 = self.CalcularCoeficientes() # Valor seleccionado de X.
        # Y_prediccion = B0 + (B1 * self.ValorX) # Predicción de Y/Regresion Lineal.
        Y_prediccion = B0 + (B1 * X_prediccion)
        return Y_prediccion

class CalcularCoeficienteCorrelacion():
    def __init__(self, MatDiscretas):
        self.MatDiscretas = MatDiscretas
        self.SSXY, self.SSXX, self.SSYY = MatDiscretas.ObtenerDatosCalcularCoeficienteCorrelacion()
    def CoeficienteCorrelacion(self):
        Correlacion_xy = self.SSXY / (self.SSXX * self.SSYY)**0.5
        return Correlacion_xy
    
class CalcularCoeficienteDeterminacion():
    def __init__(self, MatDiscretas, LinearRegresion, CoeficienteXY, X_Prediccion):
        self.MatDiscretas = MatDiscretas
        self.LinearRegresion = LinearRegresion
        self.Y, self.MediaY = MatDiscretas.ObtenerDatosCalcularCoeficienteDeterminacion()
        self.Y_Pred = LinearRegresion.predecir_y(X_Prediccion)
        self.CorrelationXY = CoeficienteXY.CoeficienteCorrelacion()
    def CoeficienteDeterminacion01(self):
        R_cuadrado_1 = self.CorrelationXY ** 2 # Coeficiente de correlación al cuadrado.
        return R_cuadrado_1
    def CoeficienteDeterminacion02(self):
        self.SSE = sum((y - self.Y_Pred) ** 2 for y in self.Y) # Varianza Residual del Modelo. Sum of Squares Error.
        self.SST = sum((y - self.MediaY) ** 2 for y in self.Y) # Varianza Total de la Variable Dependiente. Total Sum of Squares.
        R_cuadrado_2 = 1 - (self.SSE / self.SST)
        return R_cuadrado_2

print("\nPredicción con DataSet Original")
# Primer DataSet
X = [23, 26, 30, 34, 43, 48, 52, 57, 58]
Y = [651, 762, 856, 1063, 1190, 1298, 1421, 1440, 1518]
# Crear instancia de la clase MatematicasDiscretas(), vamos a mandarle el dataset para que haga algunos calculos.
PrimeraPrueba_PrimerosCalculos = MatematicasDiscretas(X,Y)
# Crear instancia de la clase RegresionLineal(), le pasamos la instancia anterior, así le vamos a pasar varios calculos necesarios.
regresion = RegresionLineal(PrimeraPrueba_PrimerosCalculos)
# Calculamos los coeficientes de la regresión lineal.
B0, B1 = regresion.CalcularCoeficientes()
print("B0:", B0)
print("B1:", B1)
# Asignamos un valor de X para predecir Y.
X_for_Pred = X[0]
# Predecimos el valor de Y.
Y_prediccion = regresion.predecir_y(X_for_Pred)
# Mostramos el resultado de la regresión lineal.
print(f"Ecuacion de Regresion: y = {B0} + ({B1} * {X_for_Pred}) = {Y_prediccion}")
#Crear instancia de la clase CalcularCoeficienteCorrelacion(), le pasamos la instancia de los calculos porque vamos a necesitar varios.
coeficiente_correlacion = CalcularCoeficienteCorrelacion(PrimeraPrueba_PrimerosCalculos)
# Calculamos el coeficiente de correlación.
CorrelacionXY = coeficiente_correlacion.CoeficienteCorrelacion()
print(f"Coeficiente de Correlacion: {CorrelacionXY}")
#Crear instancia de la clase CalcularCoeficienteDeterminacion(), le pasamos la instancia de los calculos, la regresion y correlación, porque ocupamos varios datos.
coeficiente_determinacion = CalcularCoeficienteDeterminacion(PrimeraPrueba_PrimerosCalculos,regresion,coeficiente_correlacion, X_for_Pred)
# Calculamos el coeficiente de determinación en su primera forma.
RSquare1 = coeficiente_determinacion.CoeficienteDeterminacion01()
print(f"Coeficiente de Determinacion (R * R): {CorrelacionXY} * {CorrelacionXY} = {RSquare1}")
# Calculamos el coeficiente de determinación en su segunda forma.
RSquare2 = coeficiente_determinacion.CoeficienteDeterminacion02()
print(f"Coeficiente de Determinacion (1 - (SSE/SST)): {1} - ({coeficiente_determinacion.SSE}/{coeficiente_determinacion.SST}) = {RSquare2}")
print("")

# Segundo DataSet
X1 = [10, 12, 14, 16, 18, 20, 22, 24, 26]
Y1 = [20, 24, 28, 32, 36, 40, 44, 48, 52]
contador = 1
for x in X1[:5]:
    X_for_Pred = x

    SegundaPrueba_Calculos = MatematicasDiscretas(X1,Y1)
    regresion = RegresionLineal(SegundaPrueba_Calculos)
    coeficiente_correlacion = CalcularCoeficienteCorrelacion(SegundaPrueba_Calculos)
    coeficiente_determinacion = CalcularCoeficienteDeterminacion(SegundaPrueba_Calculos,regresion,coeficiente_correlacion, X_for_Pred)

    B0, B1 = regresion.CalcularCoeficientes()
    Y_prediccion = regresion.predecir_y(X_for_Pred)
    CorrelacionXY = coeficiente_correlacion.CoeficienteCorrelacion()
    RSquare1 = coeficiente_determinacion.CoeficienteDeterminacion01()
    RSquare2 = coeficiente_determinacion.CoeficienteDeterminacion02()

    print(f"Predicción No.{contador}")
    print("B0:", B0)
    print("B1:", B1)
    print(f"Ecuacion de Regresion: y = {B0} + ({B1} * {X_for_Pred}) = {Y_prediccion}")
    print(f"Coeficiente de Correlacion: {CorrelacionXY}")
    print(f"Coeficiente de Determinacion (R * R): {CorrelacionXY} * {CorrelacionXY} = {RSquare1}")
    print(f"Coeficiente de Determinacion (1 - (SSE/SST)): {1} - ({coeficiente_determinacion.SSE}/{coeficiente_determinacion.SST}) = {RSquare2}")

    contador += 1
    
    print("")