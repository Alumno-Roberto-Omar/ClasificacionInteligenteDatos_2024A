#include<iostream>
#include<vector>
#include<cmath>

using namespace std;

struct DatosRegresionLineal {
    float SumatoriaXCuadrado;
    float SumatoriaXY;
};

struct DatosCoeficienteCorrelacion{
    float SS_XY;
    float SS_XX;
    float SS_YY;
};

struct DatosCoeficienteDeterminacion{
    float Media_Y;
};

struct ResultadosCoeficientesRegresion {
    float B0;
    float B1;
};

struct ResultadoPrediccionYRegresion{
    float Prediccion_Y;
};

struct ResultadoCoeficienteCorrelacion{
    float Correlacion_XY;
};

struct ResultadoCoeficienteDeterminacion01{
    float Determinacion01;
};

struct ResultadoCoeficienteDeterminacion02{
    float Determinacion02;
};

class MatematicasDiscretas{
    private:
        //
    public:
        std::vector<float> X;
        std::vector<float> Y;
        int N;
        float SumatoriaX;
        float SumatoriaY;
        float MediaY;
        MatematicasDiscretas(std::vector<float> X, std::vector<float> Y);
        DatosRegresionLineal ObtenerDatosCalcularRegresionLineal();
        DatosCoeficienteCorrelacion ObtenerDatosCalcularCoeficienteCorrelacion();
        DatosCoeficienteDeterminacion ObtenerDatosCalcularCoeficienteDeterminacion();
};

MatematicasDiscretas::MatematicasDiscretas(std::vector<float> X, std::vector<float> Y) : X(X), Y(Y), N(X.size()){
    this->SumatoriaX = 0;
    this->SumatoriaY = 0;
    for(int i=0; i<N; i++){
        SumatoriaX += X[i];
        SumatoriaY += Y[i];
    }
    // this->MediaY = SumatoriaY/N;
    MediaY = SumatoriaY/N;
}

DatosRegresionLineal MatematicasDiscretas::ObtenerDatosCalcularRegresionLineal(){
    float SumatoriaXCuadrado = 0;
    float SumatoriaXY = 0;
    for (int i = 0; i < N; ++i) {
        SumatoriaXCuadrado += X[i] * X[i];
        SumatoriaXY += X[i] * Y[i];
    }
    DatosRegresionLineal datos;
    datos.SumatoriaXCuadrado = SumatoriaXCuadrado;
    datos.SumatoriaXY = SumatoriaXY;
    return datos;
}

DatosCoeficienteCorrelacion MatematicasDiscretas::ObtenerDatosCalcularCoeficienteCorrelacion(){
    float MediaX = SumatoriaX/N;
    float SS_XY = 0, SS_XX = 0, SS_YY = 0;
    for (int i = 0; i < N; ++i) {
        SS_XY += (X[i] - MediaX) * (Y[i] - MediaY);
        SS_XX += (X[i] - MediaX) * (X[i] - MediaX);
        SS_YY += (Y[i] - MediaY) * (Y[i] - MediaY);
    }
    DatosCoeficienteCorrelacion datos;
    datos.SS_XY = SS_XY;
    datos.SS_XX = SS_XX;
    datos.SS_YY = SS_YY;
    return datos;
}

DatosCoeficienteDeterminacion MatematicasDiscretas::ObtenerDatosCalcularCoeficienteDeterminacion(){
    DatosCoeficienteDeterminacion datos;
    datos.Media_Y = MediaY;
    return datos;
}

class RegresionLineal{
    private:
        MatematicasDiscretas MatDiscretas;
        //std::vector<float> X;
        int N;
        float SumX, SumY, SumXCuad, SumXY;
        DatosRegresionLineal datos;
        ResultadosCoeficientesRegresion resultados;
    public:
        RegresionLineal(MatematicasDiscretas MatDiscretas);
        ResultadosCoeficientesRegresion CalcularCoeficientes();
        ResultadoPrediccionYRegresion Predecir_Y(float& X_prediccion);
};

RegresionLineal::RegresionLineal(MatematicasDiscretas MatDiscretas):MatDiscretas(MatDiscretas){
    //X = MatDiscretas.X;
    N = MatDiscretas.N;
    SumX = MatDiscretas.SumatoriaX;
    SumY = MatDiscretas.SumatoriaY;
    datos = MatDiscretas.ObtenerDatosCalcularRegresionLineal();
    SumXCuad = datos.SumatoriaXCuadrado;
    SumXY = datos.SumatoriaXY;
}

ResultadosCoeficientesRegresion RegresionLineal::CalcularCoeficientes(){
    float B0 = 0;
    float B1 = 0;
    B0 = (SumY * SumXCuad - SumX * SumXY)/(N * SumXCuad - (SumX * SumX));
    B1 = (N * SumXY - SumX * SumY)/(N * SumXCuad - (SumX * SumX));
    ResultadosCoeficientesRegresion resultados;
    resultados.B0 = B0;
    resultados.B1 = B1;
    return resultados;
}

ResultadoPrediccionYRegresion RegresionLineal::Predecir_Y(float& X_prediccion){
    ResultadosCoeficientesRegresion resultados;
    resultados = CalcularCoeficientes();
    float B0 = resultados.B0;
    float B1 = resultados.B1;
    float Y_Prediccion = 0;
    Y_Prediccion = B0 + (B1 * X_prediccion);
    ResultadoPrediccionYRegresion prediccion;
    prediccion.Prediccion_Y = Y_Prediccion;
    return prediccion;
}

class CalcularCoeficienteCorrelacion{
    private:
        MatematicasDiscretas MatDiscretas;
        DatosCoeficienteCorrelacion datos;
        float SS_XY = 0, SS_XX = 0, SS_YY = 0;
    public:
        CalcularCoeficienteCorrelacion(MatematicasDiscretas MatDiscretas);
        ResultadoCoeficienteCorrelacion CoeficienteCorrelacion();
};

CalcularCoeficienteCorrelacion::CalcularCoeficienteCorrelacion(MatematicasDiscretas MatDiscretas):MatDiscretas(MatDiscretas){
    datos = MatDiscretas.ObtenerDatosCalcularCoeficienteCorrelacion();
    SS_XY = datos.SS_XY;
    SS_XX = datos.SS_XX;
    SS_YY = datos.SS_YY;
}

ResultadoCoeficienteCorrelacion CalcularCoeficienteCorrelacion::CoeficienteCorrelacion(){
    float Correlacion_xy = 0;
    Correlacion_xy = SS_XY / (sqrt(SS_XX) * sqrt(SS_YY));
    ResultadoCoeficienteCorrelacion resultado;
    resultado.Correlacion_XY = Correlacion_xy;
    return resultado;
}

class CalcularCoeficienteDeterminacion{
    private:
        std::vector<float> Y;
        MatematicasDiscretas MatDiscretas;
        RegresionLineal& Regresion;
        CalcularCoeficienteCorrelacion& Correlacion;
        ResultadoCoeficienteCorrelacion dato_correlacion;
        DatosCoeficienteDeterminacion datos_determinacion;
        ResultadoPrediccionYRegresion PrediccionY;
        float N, MediaY, CorrelacionXY, X_Prediccion, SSE, SST, Y_Prediccion;
    public:
        CalcularCoeficienteDeterminacion(std::vector<float> Y, MatematicasDiscretas MatDiscretas, RegresionLineal& Regresion, CalcularCoeficienteCorrelacion& Correlacion, float& X_prediccion);
        ResultadoCoeficienteDeterminacion01 CoeficienteDeterminacion01();
        ResultadoCoeficienteDeterminacion02 CoeficienteDeterminacion02();
};

CalcularCoeficienteDeterminacion::CalcularCoeficienteDeterminacion(std::vector<float> Y, MatematicasDiscretas MatDiscretas, RegresionLineal& Regresion, CalcularCoeficienteCorrelacion& Correlacion, float& X_prediccion):Y(Y),MatDiscretas(MatDiscretas), Regresion(Regresion), Correlacion(Correlacion){
    // Y = MatDiscretas.Y;
    N = MatDiscretas.N;
    dato_correlacion = Correlacion.CoeficienteCorrelacion();
    CorrelacionXY = dato_correlacion.Correlacion_XY;
    datos_determinacion = MatDiscretas.ObtenerDatosCalcularCoeficienteDeterminacion();
    MediaY = datos_determinacion.Media_Y;
    PrediccionY = Regresion.Predecir_Y(X_prediccion);
    Y_Prediccion = PrediccionY.Prediccion_Y;
}

ResultadoCoeficienteDeterminacion01 CalcularCoeficienteDeterminacion::CoeficienteDeterminacion01(){ 
    float R_Cuadrado_1 = 0;
    R_Cuadrado_1 = CorrelacionXY * CorrelacionXY;
    ResultadoCoeficienteDeterminacion01 resultado;
    resultado.Determinacion01 = R_Cuadrado_1;
    return resultado;
}

ResultadoCoeficienteDeterminacion02 CalcularCoeficienteDeterminacion::CoeficienteDeterminacion02(){
    float R_Cuadrado_2 = 0;
    SSE = 0, SST = 0;
    for (int i = 0; i < N; i++)
    {
        SSE += (Y[i] - Y_Prediccion) * (Y[i] - Y_Prediccion);
        SST += (Y[i] - MediaY) * (Y[i] - MediaY);
    }
    R_Cuadrado_2 = 1 - (SSE/SST);
    ResultadoCoeficienteDeterminacion02 resultado;
    resultado.Determinacion02 = R_Cuadrado_2;
    return resultado;
}

int main()
{
    std::vector<float> X = {23, 26, 30, 34, 43, 48, 52, 57, 58};
    std::vector<float> Y = {651, 762, 856, 1063, 1190, 1298, 1421, 1440, 1518};
    /*int N;
    float SumX, SumY, SumXCuad, SumXY;*/
    MatematicasDiscretas EnvioDatos(X, Y);
    RegresionLineal Regresion(EnvioDatos);
    ResultadosCoeficientesRegresion resultados_regresion = Regresion.CalcularCoeficientes();
    float X_for_Pred = X[0];
    ResultadoPrediccionYRegresion resultado_prediccion_y = Regresion.Predecir_Y(X_for_Pred);
    float B0 = resultados_regresion.B0;
    float B1 = resultados_regresion.B1;
    float PrediccionY = resultado_prediccion_y.Prediccion_Y;
    cout<<"B0: "<<B0<<endl;
    cout<<"B1: "<<B1<<endl;
    cout<<"Y: "<<PrediccionY<<endl;
    CalcularCoeficienteCorrelacion Correlacion(EnvioDatos);
    ResultadoCoeficienteCorrelacion resultado_correlacion = Correlacion.CoeficienteCorrelacion();
    float CorrelacionXY = resultado_correlacion.Correlacion_XY;
    cout<<"Coeficiente Correlacion: "<<CorrelacionXY<<endl;
    CalcularCoeficienteDeterminacion Determinacion(Y, EnvioDatos, Regresion, Correlacion, X_for_Pred);
    ResultadoCoeficienteDeterminacion01 resultado_determinacion_01 = Determinacion.CoeficienteDeterminacion01();
    float Determinacion01 = resultado_determinacion_01.Determinacion01;
    cout<<"Coeficiente Determinacion: "<<Determinacion01<<endl;
    ResultadoCoeficienteDeterminacion02 resultado_determinacion_02 = Determinacion.CoeficienteDeterminacion02();
    float Determinacion02 = resultado_determinacion_02.Determinacion02;
    cout<<"Coeficiente Determinacion: "<<Determinacion02<<endl;
    cout<<endl;

    std::vector<float> X1 = {10, 12, 14, 16, 18, 20, 22, 24, 26};
    std::vector<float> Y1 = {20, 24, 28, 32, 36, 40, 44, 48, 52};

    // int N = X1.size();

    for (int i = 0; i < 5; i++){
        float X_para_Pred = X1[i];

        MatematicasDiscretas DatosPrueba(X1, Y1);
        RegresionLineal Regresion_Lineal(DatosPrueba);
        ResultadosCoeficientesRegresion resultados_de_regresion = Regresion_Lineal.CalcularCoeficientes();
        ResultadoPrediccionYRegresion resultado_de_prediccion_y = Regresion_Lineal.Predecir_Y(X_para_Pred);
        CalcularCoeficienteCorrelacion Correlacion_de_XY(DatosPrueba);
        ResultadoCoeficienteCorrelacion resultado_de_correlacion = Correlacion_de_XY.CoeficienteCorrelacion();
        CalcularCoeficienteDeterminacion CoeficienteDeterminacion(Y1, DatosPrueba, Regresion_Lineal, Correlacion_de_XY, X_para_Pred);
        ResultadoCoeficienteDeterminacion01 resultado_de_determinacion_01 = CoeficienteDeterminacion.CoeficienteDeterminacion01();
        ResultadoCoeficienteDeterminacion02 resultado_de_determinacion_02 = CoeficienteDeterminacion.CoeficienteDeterminacion02();

        float B_0 = resultados_de_regresion.B0;
        float B_1 = resultados_de_regresion.B1;
        float Prediccion_de_Y = resultado_de_prediccion_y.Prediccion_Y;
        float Correlacion_para_XY = resultado_de_correlacion.Correlacion_XY;
        float CoeficienteDeterminacion01 = resultado_de_determinacion_01.Determinacion01;
        float CoeficienteDeterminacion02 = resultado_de_determinacion_02.Determinacion02;

        cout<<"Prediccion "<<i+1<<endl;
        cout<<"B0: "<<B_0<<endl;
        cout<<"B1: "<<B_1<<endl;
        cout<<"Y: "<<Prediccion_de_Y<<endl;
        cout<<"Coeficiente Correlacion: "<<Correlacion_para_XY<<endl;
        cout<<"Coeficiente Determinacion: "<<CoeficienteDeterminacion01<<endl;
        cout<<"Coeficiente Determinacion: "<<CoeficienteDeterminacion02<<endl;

        cout<<endl;
    }

    return 0;
}
