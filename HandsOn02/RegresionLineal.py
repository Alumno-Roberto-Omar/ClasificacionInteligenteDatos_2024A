# Regresión Lineal Simple (Incluye Coeficiente de Correlación y Determinación)
    
class MatematicasDiscretas():
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.N = len(X)
        self.SumatoriaX = sum(X)
        self.SumatoriaY = sum(Y)
        self.MediaY = self.SumatoriaY/self.N
    def ObtenerDatosCalcularRegresionLineal(self):
        SumatoriaXCuadrado = sum(x ** 2 for x in self.X)
        SumatoriaXY = sum(x * y for x, y in zip(self.X, self.Y))
        return self.X, self.N, self.SumatoriaX, self.SumatoriaY, SumatoriaXCuadrado, SumatoriaXY
    def ObtenerDatosCalcularCoeficienteCorrelacion(self):
        MediaX = self.SumatoriaX/self.N
        SS_XY = sum((x - MediaX) * (y - self.MediaY) for x, y in zip(self.X, self.Y))
        SS_XX = sum((x - MediaX) ** 2 for x in self.X)
        SS_YY = sum((y - self.MediaY) ** 2 for y in self.Y)
        return SS_XY, SS_XX, SS_YY
    def ObtenerDatosCalcularSumaCuadrados(self):
        return self.X, self.Y, self.MediaY

class RegresionLineal():
    def __init__(self, MatDiscretas):
        self.MatDiscretas = MatDiscretas
        self.X, self.N, self.SumX, self.SumY, self.SumXCuad, self.SumXY = MatDiscretas.ObtenerDatosCalcularRegresionLineal()
    def CalcularCoeficientes(self):
        B0 = (self.SumY * self.SumXCuad - self.SumX * self.SumXY) / (self.N * self.SumXCuad - self.SumX ** 2) # Constante
        B1 = (self.N * self.SumXY - self.SumX * self.SumY) / (self.N * self.SumXCuad - self.SumX ** 2) # Coeficiente
        return B0, B1
    def Predecir_Y(self, X_prediccion):
        B0, B1 = self.CalcularCoeficientes()
        Y_prediccion = B0 + (B1 * X_prediccion)
        return Y_prediccion

class CoeficienteCorrelacion():
    def __init__(self, MatDiscretas):
        self.MatDiscretas = MatDiscretas
        self.SSXY, self.SSXX, self.SSYY = MatDiscretas.ObtenerDatosCalcularCoeficienteCorrelacion()
    def CalcularCorrelacion(self):
        Correlacion_xy = self.SSXY / (self.SSXX * self.SSYY)**0.5
        return Correlacion_xy

class CoeficienteDeterminacion_CorrelacionCuadrada():
    def __init__(self, Correlacion):
        self.Correlacion = Correlacion
        self.CorrelacionXY = Correlacion.CalcularCorrelacion()
    def CalcularDeterminacion_CorrelacionCuadrada():
        R_Cuadrada = (CorrelacionXY) ** 2
        return R_Cuadrada

class CalcularSumasDeCuadrados():
    def __init__(self, MatDiscretas, RegresionLineal):
        self.MatDiscretas = MatDiscretas
        self.RegresionLineal = RegresionLineal
        self.X, self.Y, self.MediaY = MatDiscretas.ObtenerDatosCalcularSumaCuadrados()
        self.Prediccion_Y = [RegresionLineal.Predecir_Y(x) for x in self.X]
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
    def CalcularDeterminacion_02(self):
        R_Cuadrada_3 = self.SSR / self.SST
        return R_Cuadrada_3

# Primer DataSet
# X = [23, 26, 30, 34, 43, 48, 52, 57, 58]
X = [1,2,3,4,5,6,7,8,9]
# Y = [651, 762, 856, 1063, 1190, 1298, 1421, 1440, 1518]
Y = [3,6,9,12,15,18,21,24,27]
X_Exogeno = [15,35,55,75,95]

DatosPrimeraPrueba = MatematicasDiscretas(X,Y) # Crear instancia de la clase MatematicasDiscretas(), vamos a mandarle el DataSet.
RegresionLinealSimple = RegresionLineal(DatosPrimeraPrueba) # Crear instancia de la clase RegresionLineal(), le pasamos la instancia anterior.
Correlacion = CoeficienteCorrelacion(DatosPrimeraPrueba) # Crear instancia de la clase CoeficienteCorrelacion(), le pasamos los datos que calculamos en MatDiscretas.
SumaDeCuadrados = CalcularSumasDeCuadrados(DatosPrimeraPrueba, RegresionLinealSimple) # Crear instancia de la clase CalcularSumaDeCuadrados(), le pasamos los datos de MatDiscretas y de la Regresion Lineal.
Determinacion = CoeficienteDeterminacion_SumaDeCuadrados(SumaDeCuadrados) # Determinacion 

X_para_Predicir = X[0] # Voy a usar el primer dato de X para la primera predicción.
B0, B1 = RegresionLinealSimple.CalcularCoeficientes() # Calculamos los coeficientes de regresión.
Prediccion_de_Y = RegresionLinealSimple.Predecir_Y(X_para_Predicir) # Calcular la predicción para Y y le paso el valor de X para esto.
CorrelacionXY = Correlacion.CalcularCorrelacion() # Calculamos el coeficiente de correlacion
DeterminacionY = Determinacion.CalcularDeterminacion_01() # Calculamos el coeficiente de determinación.

print("\nRegresión Lineal Simple - Y = B0 + (B1 * X)")

print("\n1.1. Predicción con DataSet Original")
print(f"B0: {round(B0,4)}  B1: {round(B1,4)}")
print(f"Y: {round(B0,4)} + ({round(B1,4)} * {round(X_para_Predicir,4)}) = {round(Prediccion_de_Y,4)}")
print("Coeficiente de Correlación (R): ", round(CorrelacionXY,4))
print("Coeficiente de Determinación (R2): ", round(DeterminacionY,4))

print("\n1.2. Predicciones con Valores Externos al DataSet Original")
conteo = 1
for x in X_Exogeno[:5]:
    X_Prediccion = x
    Y_Prediccion = RegresionLinealSimple.Predecir_Y(X_Prediccion)
    print(f"{conteo}. Y: {round(B0,4)} + ({round(B1,4)} * {round(X_Prediccion,4)}) = {round(Y_Prediccion,4)}")
    conteo += 1

X1 = [108, 115, 106, 97, 95, 91, 97, 83, 83, 78, 54, 67, 56, 53, 61, 115, 81, 78, 30, 45, 99, 32, 25, 28, 90, 89]
Y1 = [95, 96, 95, 97, 93, 94, 95, 93, 92, 86, 73, 80, 65, 69, 77, 96, 87, 89, 60, 63, 95, 61, 55, 56, 94, 93]

DatosSegundaPrueba = MatematicasDiscretas(X1,Y1)
Regresion_Lineal = RegresionLineal(DatosSegundaPrueba)
Coeficiente_Correlacion = CoeficienteCorrelacion(DatosSegundaPrueba)
SumaDeCuadrados1 = CalcularSumasDeCuadrados(DatosSegundaPrueba, Regresion_Lineal)
DeterminacionCuadrados = CoeficienteDeterminacion_SumaDeCuadrados(SumaDeCuadrados1)

B_0, B_1 = Regresion_Lineal.CalcularCoeficientes()
CorrXY = Coeficiente_Correlacion.CalcularCorrelacion()
Det = DeterminacionCuadrados.CalcularDeterminacion_01()

print("\n2. Predicciones con DataSet Secundario")
print(f"B0: {round(B_0,4)}  B1: {round(B_1,4)}")
print("Coeficiente de Correlación (R): ", round(CorrXY,4))
print("Coeficiente de Determinación (R2): ", round(Det,4))
print("")
contador = 1
for x in X1[:5]:
    X_for_Pred = x
    Y_Pred = Regresion_Lineal.Predecir_Y(X_for_Pred)
    print(f"{contador}. Y: {round(B_0,4)} + ({round(B_1,4)} * {round(X_for_Pred,4)}) = {round(Y_Pred,4)}")
    contador += 1

print("")
