import numpy as np
import matplotlib.pyplot as plt

def coeficiente_estimado(x, y):
    # cantidad de pares a observar
    n = np.size(x)
 
    # Medias de los vectores x e y.
    media_x = np.mean(x)
    media_y = np.mean(y)
 
    # Calculando las desviaciones cruzadas y la desviación sobre x (Suma de cuadrados) https://www.youtube.com/watch?v=eq90Ue6Sx5o
    # la sumatoria de x por y menos n por la media de x por la media de y sobre
    # la sumatoria de x cuadrada menos n por la media de x al cuadrado.     
    # Calculando los coeficientes de regresion.
    b_1 = (np.sum(y*x) - n*media_y*media_x) / ( np.sum(x*x) - n*media_x*media_x )
    b_0 = media_y - b_1*media_x
 
    return (b_0, b_1)
  
def coeficiente_estimado2(x, y):
    # cantidad de pares a observar
    n = np.size(x)
 
    # Medias de los vectores x e y.
    media_x = np.mean(x)
    media_y = np.mean(y)

    # Calculando la suma de residuos al cuadrado https://rpubs.com/Cristina_Gil/Regresion_Lineal_Simple
    # la sumatoria de x menos la media de x por y menos la media de y sobre
    # la sumatoria de x menos la media de x al cuadrado     
    # Calculando los coeficientes de regresion.

    sumxmedxymedy = 0;
    sumxsqrd = 0;
    for i in range(0,n):
      sumxmedxymedy += (x[i] - media_x)*(y[i]-media_y)
      sumxsqrd += ((x[i]-media_x)*(x[i]-media_x))

    b_1 = sumxmedxymedy / sumxsqrd
    b_0 = media_y - b_1*media_x

    return (b_0, b_1)
 
def coeficiente_estimado3(x, y):
    # cantidad de pares a observar
    n = np.size(x)
 
    # Medias de los vectores x e y.
    media_x = np.mean(x)
    media_y = np.mean(y)

    # Calculando los coeficientes de regresion como fue visto en clase
    # la sumatoria de x por y, menos la sumatoria de x por la sumatoria de y sobre
    # la sumatoria de x al cuadrado menos el resultado de la sumatoria de x, elevado al cuadrado     

    sumxmedxymedy = 0;
    sumxsqrd = 0;
    for i in range(0,n):
      sumxmedxymedy += (x[i] - media_x)*(y[i]-media_y)
      sumxsqrd += ((x[i]-media_x)*(x[i]-media_x))

    b_1 = (n*np.sum(y*x) - np.sum(x)*np.sum(y))/ (n*np.sum(x*x) - np.sum(x)*np.sum(x))
    b_0 = (np.sum(y)- b_1 * np.sum(x))/n

    return (b_0, b_1)


def mostrar_linea_regresion(x, y, beta):
    
    plt.scatter(x, y, color = "c",
               marker = "o", s = 30)
 
    # Vector de respuesta predecido
    y_pred = beta[0] + beta[1]*x
 
    # mostrando la linea de regresion
    plt.plot(x, y_pred, color = "r")
 
    # etiquetas
    plt.xlabel('Sales')
    plt.ylabel('Advertising')

    plt.show()
 
def main():
    # Datos
    x = np.array([23, 26, 30, 34, 43, 48, 52, 57, 58])
    y = np.array([651, 762, 856, 1063, 1190, 1298, 1421, 1440, 1518])
 
    # Procesar los coeficientes estimados con la primer formula
    beta = coeficiente_estimado(x, y)
    print("Coeficientes estimados:\nb_0 = {}  \
          \nb_1 = {}".format(beta[0], beta[1]))
    mostrar_linea_regresion(x, y, beta)

    # Procesar los coeficientes estimados con la segunda formula
    beta2 = coeficiente_estimado2(x, y)
    print("Coeficientes estimados:\nb_0 = {}  \
          \nb_1 = {}".format(beta2[0], beta2[1]))
    mostrar_linea_regresion(x, y, beta2)

    # Procesar los coeficientes estimados con la segunda formula
    beta3 = coeficiente_estimado3(x, y)
    print("Coeficientes estimados:\nb_0 = {}  \
          \nb_1 = {}".format(beta3[0], beta3[1]))
    mostrar_linea_regresion(x, y, beta3)
 
if __name__ == "__main__":
    main()