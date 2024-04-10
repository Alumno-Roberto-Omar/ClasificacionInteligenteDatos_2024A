# Regresion Polinomial
# Incluye Regresión Lineal Simple, Regresión Cuadrática y Regresión Cúbica.
# También incluye el cálculo de coeficientes de correlación (R), determinación (R2) y determinación ajustada (R2_Aj).
# Además de la desviación estandar (S).

class MatematicasDiscretas():
    def __init__(self):
        self.X = [108, 115, 106, 97, 95, 91, 97, 83, 83, 78, 54, 67, 56, 53, 61, 115, 81, 78, 30, 45, 99, 32, 25, 28, 90, 89]
        self.Y = [95, 96, 95, 97, 93, 94, 95, 93, 92, 86, 73, 80, 65, 69, 77, 96, 87, 89, 60, 63, 95, 61, 55, 56, 94, 93]
        self.N = len(self.X)
        self.SumatoriaX = sum(self.X)
        self.SumatoriaY = sum(self.Y)
        self.SumatoriaX2 = sum(x ** 2 for x in self.X)
        self.SumatoriaX3 = sum(x ** 3 for x in self.X)
        self.SumatoriaX4 = sum(x ** 4 for x in self.X)
        self.SumatoriaX5 = sum(x ** 5 for x in self.X)
        self.SumatoriaX6 = sum(x ** 6 for x in self.X)
        self.SumatoriaXY = sum(x * y for x, y in zip(self.X, self.Y))
        self.SumatoriaX2Y = sum(x ** 2 * y for x, y in zip(self.X, self.Y))
        self.SumatoriaX3Y = sum(x ** 3 * y for x, y in zip(self.X, self.Y))
        self.SumatoriaX4Y = sum(x ** 4 * y for x, y in zip(self.X, self.Y))
        self.MediaX = self.SumatoriaX/self.N
        self.MediaY = self.SumatoriaY/self.N
        self.MediaX2 = self.SumatoriaX2/self.N
    def ObtenerDatosDataSet(self):
        return self.X, self.Y
    def ObtenerDatosCalcularRegresionLineal(self):
        return self.X, self.N, self.SumatoriaX, self.SumatoriaY, self.SumatoriaX2, self.SumatoriaXY
    def ObtenerDatosCalcularRegresionCuadratica(self):
        SumXX = self.SumatoriaX2 - ((self.SumatoriaX ** 2) / self.N)
        SumXY = self.SumatoriaXY - ((self.SumatoriaX * self.SumatoriaY) / self.N)
        SumXX2 = self.SumatoriaX3 - ((self.SumatoriaX2 * self.SumatoriaX) / self.N)
        SumX2X2 = self.SumatoriaX4 - (((self.SumatoriaX2)**2) / self.N)
        SumX2Y = self.SumatoriaX2Y - ((self.SumatoriaX2 * self.SumatoriaY) / self.N)
        return SumXX, SumXY, SumXX2, SumX2Y, SumX2X2, self.MediaX, self.MediaY, self.MediaX2
    def ObtenerDatosCalcularRegresionCubica(self):
        return self.N, self.SumatoriaX, self.SumatoriaY, self.SumatoriaX2, self.SumatoriaX3, self.SumatoriaX4, self.SumatoriaX5, self.SumatoriaX6, self.SumatoriaXY, self.SumatoriaX2Y, self.SumatoriaX3Y
    def ObtenerDatosCalcularCoeficienteCorrelacion(self):
        SS_XY = sum((x - self.MediaX) * (y - self.MediaY) for x, y in zip(self.X, self.Y))
        SS_XX = sum((x - self.MediaX) ** 2 for x in self.X)
        SS_YY = sum((y - self.MediaY) ** 2 for y in self.Y)
        return SS_XY, SS_XX, SS_YY
    def ObtenerDatosCalcularSumaCuadrados(self):
        return self.X, self.Y, self.MediaY
    def ObtenerDatosCalcularDeterminacionAjustada(self):
        return self.N
    def ObtenerDatosCalcularDesviacion(self):
        return self.X, self.Y, self.N

class RegresionLineal():
    def __init__(self, MatDiscretas):
        self.MatDiscretas = MatDiscretas
        self.X, self.N, self.SumX, self.SumY, self.SumXCuad, self.SumXY = MatDiscretas.ObtenerDatosCalcularRegresionLineal()
    def CalcularCoeficientes(self):
        B0 = (self.SumY * self.SumXCuad - self.SumX * self.SumXY) / (self.N * self.SumXCuad - self.SumX ** 2) # Constante
        B1 = (self.N * self.SumXY - self.SumX * self.SumY) / (self.N * self.SumXCuad - self.SumX ** 2) # Coeficiente
        return B0, B1
    def Predecir_Y(self, X_prediccion):
        B0, B1 = self.CalcularCoeficientes() # Valor seleccionado de X.
        Y_prediccion = B0 + (B1 * X_prediccion)
        return Y_prediccion

class RegresionCuadratica():
    def __init__(self, MatDiscretas):
        self.MatDiscretas = MatDiscretas
        self.SumXX, self.SumXY, self.SumXX2, self.SumX2Y, self.SumX2X2, self.MediaX, self.MediaY, self.MediaX2 = MatDiscretas.ObtenerDatosCalcularRegresionCuadratica()
    def CalcularCoeficienteCuadratico(self):
        B2 = ((self.SumX2Y * self.SumXX) - (self.SumXY * self.SumXX2)) / ((self.SumXX * self.SumX2X2) - ((self.SumXX2) ** 2))
        return B2 #c
    def CalcularCoeficienteLineal(self):
        B1 = ((self.SumXY * self.SumX2X2) - (self.SumX2Y * self.SumXX2)) / ((self.SumXX * self.SumX2X2) - ((self.SumXX2) ** 2))
        return B1 #b
    def CalcularConstante(self):
        B2 = self.CalcularCoeficienteCuadratico()
        B1 = self.CalcularCoeficienteLineal()
        B0 = self.MediaY - (B1 * self.MediaX) - (B2 * self.MediaX2)
        return B0 #a
    def Predecir_Y(self, X_Prediccion):
        B2 = self.CalcularCoeficienteCuadratico()
        B1 = self.CalcularCoeficienteLineal()
        B0 = self.CalcularConstante()
        Y_Predicha = (B2 * (X_Prediccion ** 2)) + (B1 * X_Prediccion) + B0
        return Y_Predicha
    
class RegresionCubica():
    def __init__(self, MatDiscretas):
        self.MatDiscretas = MatDiscretas
        self.N, self.SumX, self.SumY, self.SumX2, self.SumX3, self.SumX4, self.SumX5, self.SumX6, self.SumXY, self.SumX2Y, self.SumX3Y = MatDiscretas.ObtenerDatosCalcularRegresionCubica()
    def Eliminacion_Gaussiana(self,A,b):
        n = len(A)
        for i in range(n):
            # Hacer el pivote actual igual a 1
            factor = 1.0 / A[i][i]
            for j in range(i, n):
                A[i][j] *= factor
            b[i] *= factor
            # Hacer ceros en la columna i en las filas por debajo del pivote
            for k in range(i+1, n):
                factor = A[k][i]
                for j in range(i, n):
                    A[k][j] -= factor * A[i][j]
                b[k] -= factor * b[i]
        # Sustitución hacia atrás para resolver el sistema
        x = [0] * n
        for i in range(n-1, -1, -1):
            x[i] = b[i]
            for j in range(i+1, n):
                x[i] -= A[i][j] * x[j]
        return x
    def CalcularCoeficientesRegresionCubica(self):
        MatrizCoeficientes = [[self.SumX3, self.SumX2, self.SumX, self.N],
                              [self.SumX4, self.SumX3, self.SumX2, self.SumX],
                              [self.SumX5, self.SumX4, self.SumX3, self.SumX2],
                              [self.SumX6, self.SumX5, self.SumX4, self.SumX3]]
        VectorTermInd = [self.SumY, self.SumXY, self.SumX2Y, self.SumX3Y]
        return self.Eliminacion_Gaussiana(MatrizCoeficientes, VectorTermInd)
    def Predecir_Y(self, X_Prediccion):
        B3, B2, B1, B0 = self.CalcularCoeficientesRegresionCubica()
        Y_Predicha = (B3 * (X_Prediccion ** 3)) + (B2 * (X_Prediccion ** 2)) + (B1 * X_Prediccion) + B0
        return Y_Predicha
    
class CoeficienteCorrelacion():
    def __init__(self, MatDiscretas):
        self.MatDiscretas = MatDiscretas
        self.SSXY, self.SSXX, self.SSYY = MatDiscretas.ObtenerDatosCalcularCoeficienteCorrelacion()
    def CalcularCoeficienteCorrelacion(self):
        Correlacion_XY = self.SSXY / (self.SSXX * self.SSYY)**0.5
        return Correlacion_XY

class CalcularSumasCuadrados():
    def __init__(self, MatDiscretas, Regresion):
        self.MatDiscretas = MatDiscretas
        self.Regresion = Regresion
        self.X, self.Y, self.MediaY = MatDiscretas.ObtenerDatosCalcularSumaCuadrados()
        self.Prediccion_Y = [Regresion.Predecir_Y(x) for x in self.X]
    def SumSquaresError(self):
        SSE = sum((y - pred_y) ** 2 for y, pred_y in zip(self.Y, self.Prediccion_Y))
        return SSE
    def SumSquaresTotal(self):
        SST = sum((y - self.MediaY) ** 2 for y in self.Y)
        return SST
    def SumSquaresRegresion(self):
        SSR = sum((pred_y - self.MediaY) ** 2 for pred_y in self.Prediccion_Y)
        return SSR

class CoeficienteDeterminacion_SumaDeCuadrados():
    def __init__(self, SumSquares):
        self.SumSquares = SumSquares
        self.SSE = SumSquares.SumSquaresError()
        self.SST = SumSquares.SumSquaresTotal()
        self.SSR = SumSquares.SumSquaresRegresion()
    def CalcularDeterminacion_01(self):
        R_Cuadrada_1 = 1 - (self.SSE / self.SST)
        return R_Cuadrada_1
    def CalcularDeterminacion_02(self):
        R_Cuadrada_2 = (self.SST - self.SSE) / self.SST
        return R_Cuadrada_2
    def CalcularDeterminacion_03(self):
        R_Cuadrada_3 = self.SSR / self.SST
        return R_Cuadrada_3
    
class CoeficienteDeterminacion_Ajustado():
    def __init__(self, MatDiscretas, Determinacion):
        self.MatDiscretas = MatDiscretas
        self.Determinacion = Determinacion
        self.N = MatDiscretas.ObtenerDatosCalcularDeterminacionAjustada()
        self.R2 = Determinacion.CalcularDeterminacion_01()
    def CalcularDeterminacionAjustada(self):
        k = 1
        R_Ajustada = 1 - ((1-self.R2)*(self.N-1)/(self.N-k-1))
        return R_Ajustada
    
class DesviacionEstandar():
    def __init__(self, MatDiscretas, Regresion):
        self.MatDiscretas = MatDiscretas
        self.Regresion = Regresion
        self.X, self.Y, self.N = MatDiscretas.ObtenerDatosCalcularDesviacion()
        self.Y_Prediccion = [Regresion.Predecir_Y(x) for x in self.X]
    def CalcularDesviacion(self):
        SumatoriaErrores = sum((y - pred_y) for y, pred_y in zip(self.Y, self.Y_Prediccion))
        MediaErrores = SumatoriaErrores/self.N
        ErrorSquared = sum(((y - pred_y) - MediaErrores) ** 2 for y, pred_y in zip(self.Y, self.Y_Prediccion))
        Desviacion = (ErrorSquared/(self.N - 2))**0.5
        return Desviacion

X_Prediccion_Exogenas = [13,23,33]
DatosPrueba = MatematicasDiscretas()
Regresion_Lineal = RegresionLineal(DatosPrueba)
Regresion_Cuadratica = RegresionCuadratica(DatosPrueba)
Regresion_Cubica = RegresionCubica(DatosPrueba)
Correlacion = CoeficienteCorrelacion(DatosPrueba)
SumaCuadrados_Lineal = CalcularSumasCuadrados(DatosPrueba, Regresion_Lineal)
SumaCuadrados_Cuadratica = CalcularSumasCuadrados(DatosPrueba, Regresion_Cuadratica)
SumaCuadrados_Cubica = CalcularSumasCuadrados(DatosPrueba, Regresion_Cubica)
Determinacion_Lineal = CoeficienteDeterminacion_SumaDeCuadrados(SumaCuadrados_Lineal)
Determinacion_Cuadratica = CoeficienteDeterminacion_SumaDeCuadrados(SumaCuadrados_Cuadratica)
Determinacion_Cubica = CoeficienteDeterminacion_SumaDeCuadrados(SumaCuadrados_Cubica)
DetAjustada_Lineal = CoeficienteDeterminacion_Ajustado(DatosPrueba, Determinacion_Lineal)
DetAjustada_Cuadratica = CoeficienteDeterminacion_Ajustado(DatosPrueba, Determinacion_Cuadratica)
DetAjustada_Cubico = CoeficienteDeterminacion_Ajustado(DatosPrueba, Determinacion_Cubica)
Desviacion_Lineal = DesviacionEstandar(DatosPrueba, Regresion_Lineal)
Desviacion_Cuadratica = DesviacionEstandar(DatosPrueba, Regresion_Cuadratica)
Desviacion_Cubica = DesviacionEstandar(DatosPrueba, Regresion_Cubica)

X,Y = DatosPrueba.ObtenerDatosDataSet()
B0_Lineal, B1_Lineal = Regresion_Lineal.CalcularCoeficientes()
B2_Cuad = Regresion_Cuadratica.CalcularCoeficienteCuadratico()
B1_Cuad = Regresion_Cuadratica.CalcularCoeficienteLineal()
B0_Cuad = Regresion_Cuadratica.CalcularConstante()
B3_Cub, B2_Cub, B1_Cub, B0_Cub = Regresion_Cubica.CalcularCoeficientesRegresionCubica()
CorrelacionXY = Correlacion.CalcularCoeficienteCorrelacion()
DetLineal = Determinacion_Lineal.CalcularDeterminacion_01()
DetCuad = Determinacion_Cuadratica.CalcularDeterminacion_01()
DetCub = Determinacion_Cubica.CalcularDeterminacion_01()
DetAj_Lineal = DetAjustada_Lineal.CalcularDeterminacionAjustada()
DetAj_Cuadratica = DetAjustada_Cuadratica.CalcularDeterminacionAjustada()
DetAj_Cubica = DetAjustada_Cubico.CalcularDeterminacionAjustada()
DesvLineal = Desviacion_Lineal.CalcularDesviacion()
DesvCuad = Desviacion_Cuadratica.CalcularDesviacion()
DesvCub = Desviacion_Cubica.CalcularDesviacion()

print("\nPredicciones para Y usando Regresión Polinomial")

print("\nRegresión Lineal: B0 + B1 * X = Y")
print("Coeficiente de Correlación (R): ", round(CorrelacionXY,4))
print("Coeficiente de Determinación (R2): ", round(DetLineal,4))
print("Coeficiente de Determinación Ajustado (R2_Aj): ", round(DetAj_Lineal,4))
print("Desviación Estándar (S): ", round(DesvLineal,4))

print("\nRegresión Lineal Simple - Predicciones con Valores Conocidos")
for x in X[:3]:
    X_Prediccion = x
    Y_Pred_Lineal = Regresion_Lineal.Predecir_Y(X_Prediccion)
    print(f"{round(B0_Lineal, 4)} + ({round(B1_Lineal, 4)} * {X_Prediccion}) = {round(Y_Pred_Lineal, 4)}")

print("\nRegresión Lineal Simple - Predicciones con Variables Exógenas")
for x in X_Prediccion_Exogenas[:3]:
    X_Prediccion = x
    Y_Pred_Lineal = Regresion_Lineal.Predecir_Y(X_Prediccion)
    print(f"{round(B0_Lineal, 4)} + ({round(B1_Lineal, 4)} * {X_Prediccion}) = {round(Y_Pred_Lineal, 4)}")

print("\nRegresión Cuadrática: B0 + (B1 * X) + (B2 * X^2) = Y")
print("Coeficiente de Correlación (R): ", round(CorrelacionXY,4))
print("Coeficiente de Determinación (R2): ", round(DetCuad,4))
print("Coeficiente de Determinación Ajustado (R2_Aj): ", round(DetAj_Cuadratica,4))
print("Desviación Estándar (S): ", round(DesvCuad,4))

print("\nRegresión Cuadrática - Predicciones con Valores Conocidos")
for x in X[:3]:
    X_Prediccion = x
    Y_Pred_Cuad = Regresion_Cuadratica.Predecir_Y(X_Prediccion)
    print(f"{round(B0_Cuad, 4)} + ({round(B1_Cuad, 4)} * {X_Prediccion}) + ({round(B2_Cuad, 4)} * ({X_Prediccion}^2)) = {round(Y_Pred_Cuad, 4)}")

print("\nRegresión Cuadrática - Predicciones con Variables Exógenas")
for x in X_Prediccion_Exogenas[:3]:
    X_Prediccion = x
    Y_Pred_Cuad = Regresion_Cuadratica.Predecir_Y(X_Prediccion)
    print(f"{round(B0_Cuad, 4)} + ({round(B1_Cuad, 4)} * {X_Prediccion}) + ({round(B2_Cuad, 4)} * ({X_Prediccion}^2)) = {round(Y_Pred_Cuad, 4)}")

print("\nRegresion Cúbica: B0 + (B1 * X) + (B2 * X^2) + (B3 * X^3) = Y")
print("Coeficiente de Correlación (R): ", round(CorrelacionXY,4))
print("Coeficiente de Determinación (R2): ", round(DetCub,4))
print("Coeficiente de Determinación Ajustado (R2_Aj): ", round(DetAj_Cubica,4))
print("Desviación Estándar (S): ", round(DesvCub,4))

print("\nRegresión Cúbica - Predicciones con Valores Conocidos")
for x in X[:3]:
    X_Prediccion = x
    Y_Pred_Cub = Regresion_Cubica.Predecir_Y(X_Prediccion)
    print(f"{round(B0_Cub, 4)} + ({round(B1_Cub, 4)} * {X_Prediccion}) + ({round(B2_Cub, 4)} * ({X_Prediccion}^2)) + ({round(B3_Cub, 4)} * ({X_Prediccion}^3)) = {round(Y_Pred_Cub, 4)}")

print("\nRegresión Cúbica - Predicciones con Variables Exógenas")
for x in X_Prediccion_Exogenas[:3]:
    X_Prediccion = x
    Y_Pred_Cub = Regresion_Cubica.Predecir_Y(X_Prediccion)
    print(f"{round(B0_Cub, 4)} + ({round(B1_Cub, 4)} * {X_Prediccion}) + ({round(B2_Cub, 4)} * ({X_Prediccion}^2)) + ({round(B3_Cub, 4)} * ({X_Prediccion}^3)) = {round(Y_Pred_Cub, 4)}")

print("")
