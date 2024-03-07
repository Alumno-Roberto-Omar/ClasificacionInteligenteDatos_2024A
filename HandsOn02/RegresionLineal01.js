class MatematicasDiscretas{
    constructor(X,Y){
        this.X = X;
        this.Y = Y;
        this.N = X.length;
        this.SumatoriaX = this.X.reduce((anterior, actual) => anterior + actual, 0);
        this.SumatoriaY = this.Y.reduce((anterior, actual) => anterior + actual, 0);
        this.MediaY = this.SumatoriaY/this.N; 
    }
    ObtenerDatosCalcularRegresionLineal(){
        let SumatoriaXCuadrado = this.X.reduce((acumulador, actual) => {
            return acumulador + Math.pow(actual,2);
        }, 0);
        let SumatoriaXY = 0;
        for (let i = 0; i < this.N; i++) {
            SumatoriaXY += this.X[i] * this.Y[i];
        }
        return [this.X, this.N, this.SumatoriaX, this.SumatoriaY, SumatoriaXCuadrado, SumatoriaXY];
    }
    ObtenerDatosCalcularCoeficienteCorrelacion(){
        let MediaX = this.SumatoriaX/this.N;
        let SS_XY = 0;
        let SS_XX = 0;
        let SS_YY = 0;
        for (let i = 0; i < this.N; i++) {
            SS_XY += (this.X[i] - MediaX) * (this.Y[i] - this.MediaY);
            SS_XX += (this.X[i] - MediaX) * (this.X[i] - MediaX);
            SS_YY += (this.Y[i] - this.MediaY) * (this.Y[i] - this.MediaY);
        }
        return [SS_XY, SS_XX, SS_YY];
    }
    ObtenerDatosCalcularCoeficienteDeterminacion(){
        return [this.Y, this.MediaY, this.N]
    }
}

class RegresionLineal{
    constructor(MatDiscretas){
        this.MatDiscretas = MatDiscretas;
        let DatosRegresion = MatDiscretas.ObtenerDatosCalcularRegresionLineal();
        this.X = DatosRegresion[0];
        this.N = DatosRegresion[1];
        this.SumX = DatosRegresion[2];
        this.SumY = DatosRegresion[3];
        this.SumXCuad = DatosRegresion[4]; 
        this.SumXY = DatosRegresion[5];
    }
    CalcularCoeficientesRegresion(){
        let B0 = (this.SumY * this.SumXCuad - this.SumX * this.SumXY)/(this.N * this.SumXCuad - (this.SumX * this.SumX));
        let B1 = (this.N * this.SumXY - this.SumX * this.SumY)/(this.N * this.SumXCuad - (this.SumX * this.SumX));
        return [B0, B1];
    }
    Predecir_Y(X_Prediccion){
        let CoeficientesRegresion = this.CalcularCoeficientesRegresion();
        let B0 = CoeficientesRegresion[0];
        let B1 = CoeficientesRegresion[1];
        let Y_Prediccion = B0 + (B1 * X_Prediccion);
        return [Y_Prediccion];
    }
}

class CoeficienteCorrelacion{
    constructor(MatDiscretas){
        this.MatDiscretas = MatDiscretas;
        let DatosCorrelacion = MatDiscretas.ObtenerDatosCalcularCoeficienteCorrelacion();
        this.SSXY = DatosCorrelacion[0];
        this.SSXX = DatosCorrelacion[1];
        this.SSYY = DatosCorrelacion[2];
    }
    CalcularCorrelacion(){
        let CorrelacionXY = 0;
        CorrelacionXY = this.SSXY / Math.sqrt(this.SSXX * this.SSYY);
        return [CorrelacionXY];
    }
}

class CoeficienteDeterminacion_RCuadrado{
    constructor(Correlacion){
        this.Correlacion = Correlacion;
        let DatoCorrelacion = Correlacion.CalcularCorrelacion();
        this.CorrelacionXY = DatoCorrelacion[0];
    }
    CalcularRCuadrada(){
        let R_Cuadrada = Math.pow(this.CorrelacionXY,2);
        return [R_Cuadrada];
    }
}

class CoeficienteDeterminacion_Ajustado{
    constructor(MatDiscretas, RegresionLineal, PrediccionX){
        this.MatDiscretas = MatDiscretas;
        let DatosDeterminacion = MatDiscretas.ObtenerDatosCalcularCoeficienteDeterminacion();
        this.RegresionLineal = RegresionLineal;
        let DatosRegresion = RegresionLineal.Predecir_Y(PrediccionX);
        this.Y = DatosDeterminacion[0];
        this.MediaY = DatosDeterminacion[1];
        this.N = DatosDeterminacion[2];
        this.PrediccionY = DatosRegresion[0];
    }
    CalcularDeterminacionAjustada(){
        let SSE = 0;
        let SST = 0;
        for (let i = 0; i < this.N; i++) {
            SSE += (this.Y[i] - this.PrediccionY) * (this.Y[i] - this.PrediccionY);
            SST += (this.Y[i] - this.MediaY) * (this.Y[i] - this.MediaY);
        }
        let R_Ajustada = 1 - (SSE/SST);
        return [R_Ajustada];
    }
}

// Primer DataSet
let X = new Array(23, 26, 30, 34, 43, 48, 52, 57, 58);
let Y = new Array(651, 762, 856, 1063, 1190, 1298, 1421, 1440, 1518);
let X_for_Pred = X[0];

let MatDiscretas = new MatematicasDiscretas(X,Y);
let Regresion = new RegresionLineal(MatDiscretas);
let Correlacion = new CoeficienteCorrelacion(MatDiscretas);
let Determinacion_RCuadrada = new CoeficienteDeterminacion_RCuadrado(Correlacion);
let Determinacion_Ajustada = new CoeficienteDeterminacion_Ajustado(MatDiscretas, Regresion, X_for_Pred);

let CoeficientesRegresion = Regresion.CalcularCoeficientesRegresion();
let PrediccionY = Regresion.Predecir_Y(X_for_Pred);
let Correlacion_XY = Correlacion.CalcularCorrelacion();
let DeterminacionRCuad = Determinacion_RCuadrada.CalcularRCuadrada();
let DeterminacionAjustada = Determinacion_Ajustada.CalcularDeterminacionAjustada();

let B0 = CoeficientesRegresion[0];
let B1 = CoeficientesRegresion[1];
let Prediccion_de_Y = PrediccionY[0];
let Correlacion_de_XY = Correlacion_XY[0];
let R_Square = DeterminacionRCuad[0];
let R_Ajustado = DeterminacionAjustada[0];

console.log("Predicción con DataSet Original");
console.log("B0: ", B0);
console.log("B1: ",B1);
console.log("Y: ", Prediccion_de_Y);
console.log("R: ", Correlacion_de_XY);
console.log("R2: ", R_Square);
console.log("RA: ", R_Ajustado);
console.log(" ");

// Segundo DataSet
let X1 = new Array(10, 12, 14, 16, 18, 20, 22, 24, 26);
let Y1 = new Array(20, 24, 28, 32, 36, 40, 44, 48, 52);

for(let i=0; i<5; i++){
    let X_Pred = X1[i];

    let CalculosIniciales = new MatematicasDiscretas(X1,Y1);
    let Regresion_Lineal = new RegresionLineal(CalculosIniciales);
    let Correlacion_para_XY= new CoeficienteCorrelacion(CalculosIniciales);
    let Determination_RSquare = new CoeficienteDeterminacion_RCuadrado(Correlacion_para_XY);
    let Determination_Ajustada = new CoeficienteDeterminacion_Ajustado(CalculosIniciales, Regresion_Lineal, X_Pred);

    let Coeficientes_Regresion = Regresion_Lineal.CalcularCoeficientesRegresion();
    let Prediccion_para_Y = Regresion_Lineal.Predecir_Y(X_Pred);
    let Correlation_for_XY = Correlacion_para_XY.CalcularCorrelacion();
    let RSquare = Determination_RSquare.CalcularRCuadrada();
    let RAjustada = Determination_Ajustada.CalcularDeterminacionAjustada();

    let B0_2 = Coeficientes_Regresion[0];
    let B1_2 = Coeficientes_Regresion[1];
    let Pred_Y = Prediccion_para_Y[0];
    let Corr_XY = Correlation_for_XY[0];
    let R2 = RSquare[0];
    let RA = RAjustada[0];

    console.log("Predicción No.",i+1," con DataSet Secundario");
    console.log("B0: ", B0_2);
    console.log("B1: ",B1_2);
    console.log("Y: ", Pred_Y);
    console.log("R: ", Corr_XY);
    console.log("R2: ", R2);
    console.log("R2: ", RA);
    console.log(" ");
}