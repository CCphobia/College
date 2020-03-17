# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 1: Regresja liniowa
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

import numpy as np

from utils import polynomial


def mean_squared_error(x, y, w):
    """
    :param x: ciąg wejściowy Nx1
    :param y: ciąg wyjsciowy Nx1
    :param w: parametry modelu (M+1)x1
    :return: błąd średniokwadratowy pomiędzy wyjściami y oraz wyjściami
     uzyskanymi z wielowamiu o parametrach w dla wejść x
    """
    y_train = polynomial(x, w) #y-  Nx1

    error =  np.subtract(y, y_train) ** 2#bledy Nx1

    return np.mean(error)

def design_matrix(x_train, M):
    """
    :param x_train: ciąg treningowy Nx1
    :param M: stopień wielomianu 0,1,2,...
    :return: funkcja wylicza Design Matrix Nx(M+1) dla wielomianu rzędu M
    """
    dm = [[x_train[i][0] ** j for j in range(M+1)] for i in range(np.shape(x_train)[0])]
    return np.asarray(dm)

def least_squares(x_train, y_train, M):
    """
    :param x_train: ciąg treningowy wejśćia Nx1
    :param y_train: ciąg treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :return: funkcja zwraca krotkę (w,err), gdzie w są parametrami dopasowanego 
    wielomianu, a err to błąd średniokwadratowy dopasowania
    """
    def inverse_matrix(matrix):
        return np.linalg.inv(matrix)

    dm = design_matrix(x_train, M)
    dm_T = np.transpose(dm)
    w = (inverse_matrix((dm_T.dot(dm))).dot(dm_T)).dot(y_train)
    err = mean_squared_error(x_train, y_train, w)
    return (w, err)


def regularized_least_squares(x_train, y_train, M, regularization_lambda):
    """
    :param x_train: ciąg treningowy wejśćia Nx1
    :param y_train: ciąg treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :param regularization_lambda: parametr regularyzacji
    :return: funkcja zwraca krotkę (w,err), gdzie w są parametrami dopasowanego
    wielomianu zgodnie z kryterium z regularyzacją l2, a err to błąd 
    średniokwadratowy dopasowania
    """
    def inverse_matrix(matrix):
        return np.linalg.inv(matrix)

    dm = design_matrix(x_train, M)
    dm_T = np.transpose(dm)
    I = np.identity(np.shape(dm)[1])

    w = (inverse_matrix((dm_T.dot(dm)) + regularization_lambda * I).dot(dm_T)).dot(y_train)
    err = mean_squared_error(x_train, y_train, w)
    return w, err

def model_selection(x_train, y_train, x_val, y_val, M_values):
    """
    :param x_train: ciąg treningowy wejśćia Nx1
    :param y_train: ciąg treningowy wyjscia Nx1
    :param x_val: ciąg walidacyjny wejśćia Nx1
    :param y_val: ciąg walidacyjny wyjscia Nx1
    :param M_values: tablica stopni wielomianu, które mają byc sprawdzone
    :return: funkcja zwraca krotkę (w,train_err,val_err), gdzie w są parametrami
    modelu, ktory najlepiej generalizuje dane, tj. daje najmniejszy błąd na 
    ciągu walidacyjnym, train_err i val_err to błędy na sredniokwadratowe na 
    ciągach treningowym i walidacyjnym
    """
    w_err_tuple_to_check = [least_squares(x_train, y_train, M) for M in M_values]
    val_err_list = [mean_squared_error(x_val, y_val, w_err_tuple_to_check[i][0]) for i in M_values]
    result_triple_tuple_list = [(w_err_tuple_to_check[i][0], w_err_tuple_to_check[i][1], val_err_list[i]) for i in M_values]

    min_val_err_tuple = min(result_triple_tuple_list, key = lambda n: (n[2], n[1], n[0]))

    w = min_val_err_tuple[0]
    train_err = min_val_err_tuple[1]
    val_err = min_val_err_tuple[2]

    return w, train_err, val_err


def regularized_model_selection(x_train, y_train, x_val, y_val, M, lambda_values):
    """
    :param x_train: ciąg treningowy wejśćia Nx1
    :param y_train: ciąg treningowy wyjscia Nx1
    :param x_val: ciąg walidacyjny wejśćia Nx1
    :param y_val: ciąg walidacyjny wyjscia Nx1
    :param M: stopień wielomianu
    :param lambda_values: lista z wartościami różnych parametrów regularyzacji
    :return: funkcja zwraca krotkę (w,train_err,val_err,regularization_lambda),
    gdzie w są parametrami modelu, ktory najlepiej generalizuje dane, tj. daje
    najmniejszy błąd na ciągu walidacyjnym. Wielomian dopasowany jest wg
    kryterium z regularyzacją. train_err i val_err to błędy średniokwadratowe
    na ciągach treningowym i walidacyjnym. regularization_lambda to najlepsza
    wartość parametru regularyzacji
    """
    w_err_tuple_to_check = [regularized_least_squares(x_train, y_train, M, lam) for lam in lambda_values] #w, train_err
    val_err_list = [mean_squared_error(x_val, y_val, w_err_tuple_to_check[i][0]) for i in range(len(lambda_values))]

    possible_results_list = [(w_err_tuple_to_check[i][0], w_err_tuple_to_check[i][1], val_err_list[i], lambda_values[i]) for i in range(len(lambda_values))]

    min_tuple = min(possible_results_list, key = lambda n: (n[2], n[3], n[1], n[0]))

    w = min_tuple[0]
    train_err = min_tuple[1]
    val_err = min_tuple[2]
    regularization_lambda = min_tuple[3]

    return w, train_err, val_err, regularization_lambda