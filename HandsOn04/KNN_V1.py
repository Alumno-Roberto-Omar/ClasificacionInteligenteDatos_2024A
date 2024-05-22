import math

# Definición de la clase para el conjunto de datos
class ConjuntoDatos():
    def __init__(self):
        self.Height = [158, 158, 158, 160, 160, 163, 163, 160, 163, 160, 163, 165, 165, 165, 168, 168, 168, 170, 170, 170]
        self.Weight = [58, 59, 63, 59, 60, 60, 61, 64, 64, 61, 62, 65, 62, 63, 66, 63, 64, 68, 67, 70]
        self.Size_TShirt = ['M', 'M', 'M', 'M', 'M', 'M', 'M', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L']
    
    def obtener_datos(self):
        return list(zip(self.Height, self.Weight)), self.Size_TShirt

# Definición de la clase para calcular la distancia euclidiana
class DistanciaEuclidiana():
    @staticmethod
    def calcular(punto1, punto2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(punto1, punto2)))

# Definición de la clase para KNN
class KNN():
    def __init__(self, k=3):
        self.k = k
        self.datos = []
        self.etiquetas = []
    
    def entrenar(self, datos, etiquetas):
        self.datos = datos
        self.etiquetas = etiquetas
    
    def predecir(self, nuevo_punto):
        distancias = []
        for dato, etiqueta in zip(self.datos, self.etiquetas):
            distancia = DistanciaEuclidiana.calcular(dato, nuevo_punto)
            distancias.append((distancia, etiqueta))
        
        # Ordenar las distancias y obtener las k más cercanas
        distancias.sort(key=lambda x: x[0])
        vecinos_cercanos = distancias[:self.k]
        
        # Contar las etiquetas de los vecinos más cercanos
        conteo_etiquetas = {}
        for _, etiqueta in vecinos_cercanos:
            if etiqueta in conteo_etiquetas:
                conteo_etiquetas[etiqueta] += 1
            else:
                conteo_etiquetas[etiqueta] = 1
        
        # Determinar la etiqueta con más frecuencia
        etiqueta_predicha = max(conteo_etiquetas, key=conteo_etiquetas.get)
        
        return etiqueta_predicha

# Datos de prueba
datos_prueba = ConjuntoDatos()
datos, etiquetas = datos_prueba.obtener_datos()

# Instancia de KNN
knn = KNN(k=3)
knn.entrenar(datos, etiquetas)

# Nuevo punto a clasificar
nuevo_punto = (164, 62)
etiqueta_predicha = knn.predecir(nuevo_punto)

print(f"El nuevo punto {nuevo_punto} se clasifica como: {etiqueta_predicha}")
