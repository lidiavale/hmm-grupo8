import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from scipy.io import loadmat
from sklearn.metrics import mean_squared_error, r2_score
import random
import pandas as pd

# --- PARÂMETROS DO AG ---
POP_SIZE   = 50
N_GEN      = 100
PCROSS     = 0.8
PMUT       = 1.0 / 372    # em média 1 mutação por indivíduo
KFOLD      = 5
SEED       = 42

random.seed(SEED)
np.random.seed(SEED)

# --- FUNÇÃO DE FITNESS ---
def fitness_mask(mask, X, y):
    """
    Recebe um vetor binário 'mask' de tamanho n_features,
    aplica LinearRegression only nas colunas onde mask==1,
    retorna o RMSE médio em CV (menor é melhor).
    """
    # se não selecionou nada, penaliza fortemente
    if mask.sum() == 0:
        return np.inf

    X_sel = X[:, mask.astype(bool)]
    model = LinearRegression()

    # usamos cross_val_score com scoring negativo MSE
    kf = KFold(n_splits=KFOLD, shuffle=True, random_state=SEED)
    scores = cross_val_score(model, X_sel, y,
                             scoring='neg_mean_squared_error',
                             cv=kf)
    mse_mean = -scores.mean()
    rmse = np.sqrt(mse_mean)
    return rmse

# --- INICIALIZAÇÃO ---
def init_population(pop_size, n_features):
    return [np.random.randint(0, 2, size=n_features) for _ in range(pop_size)]

# --- TORNEIO ---
def tournament(pop, fitnesses, k=3):
    aspirants = random.sample(list(zip(pop, fitnesses)), k)
    # menor fitness (RMSE) vence
    winner = min(aspirants, key=lambda x: x[1])[0]
    return winner.copy()

# --- CROSSOVER 1‐PONTO ---
def crossover(p1, p2):
    if len(p1) < 2:
        return p1.copy(), p2.copy()
    point = random.randrange(1, len(p1))
    c1 = np.concatenate([p1[:point], p2[point:]])
    c2 = np.concatenate([p2[:point], p1[point:]])
    return c1, c2


# --- MUTAÇÃO BIT FLIP ---
def mutate(mask, pmut):
    for i in range(len(mask)):
        if random.random() < pmut:
            mask[i] = 1 - mask[i]
    return mask

# --- LOOP PRINCIPAL ---
def genetic_algorithm(X_cal, y_cal):
    n_features = X_cal.shape[1]

    pop = init_population(POP_SIZE, n_features)

    # 2) avalia fitness inicial
    fitnesses = [fitness_mask(ind, X_cal, y_cal) for ind in pop]

    best_ind, best_fit = None, float('inf')

    for gen in range(1, N_GEN+1):
        new_pop = []

        # 3) reprodução até encher nova população
        while len(new_pop) < POP_SIZE:
            # seleção
            p1 = tournament(pop, fitnesses)
            p2 = tournament(pop, fitnesses)

            # crossover
            if random.random() < PCROSS:
                c1, c2 = crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()

            # mutação
            c1 = mutate(c1, PMUT)
            c2 = mutate(c2, PMUT)

            new_pop.extend([c1, c2])

        # 4) limita tamanho
        pop = new_pop[:POP_SIZE]
        # 5) reavalia fitness
        fitnesses = [fitness_mask(ind, X_cal, y_cal) for ind in pop]

        # 6) atualiza melhor
        gen_best_idx = int(np.argmin(fitnesses))
        if fitnesses[gen_best_idx] < best_fit:
            best_fit = fitnesses[gen_best_idx]
            best_ind = pop[gen_best_idx].copy()

        # opcional: print de progresso
        print(f"Geração {gen:03d} — melhor RMSE: {best_fit:.4f} — #variáveis: {best_ind.sum()}")

    return best_ind, best_fit


if __name__ == "__main__":
    # Exemplo de carregamento (substituir pelos seus dados):
    df = pd.read_excel("Dados/2012/IDRC_Validation_set_references.xlsx")
    mat = loadmat("Dados/2012/ShootOut2012MATLAB/IDRC_Validation_set_references_Sheet1.mat")
    mat2 = loadmat("Dados/2012/ShootOut2012MATLAB/ShootOut2012MATLAB.mat")

    X_cal_calibration = mat2["inputCalibration"]   
    y_cal_calibration = mat2["targetCalibration"].ravel()

    print("X_cal_calibration shape:", X_cal_calibration.shape)
    print("y_cal_calibration shape:", y_cal_calibration.shape)
    
    X_test = mat2["inputTest"]
    y_test = mat2["targetTest"].ravel()

    X_val = mat2["inputValidation"]

    #print(" Chaves:", mat2.keys())
    #print(" . Mat Valores 2:", mat2)


    best_mask, best_rmse = genetic_algorithm(X_cal_calibration, y_cal_calibration)

    print("Máscara ótima encontrada!")
    print("RMSE (CV):", best_rmse)
    print("Número de variáveis selecionadas:", best_mask.sum())

    # Para treinar o modelo final:
    X_sel = X_cal_calibration[:, best_mask.astype(bool)]
    final_model = LinearRegression().fit(X_sel, y_cal_calibration)

    mask_bool    = best_mask.astype(bool)
    X_test_sel   = X_test[:,  mask_bool]
    X_val_sel    = X_val[:,   mask_bool]

    # 2) Gere as previsões
    y_test_pred  = final_model.predict(X_test_sel)
    y_val_pred   = final_model.predict(X_val_sel)

    rmse_test = mean_squared_error(y_test, y_test_pred)
    r2_test   = r2_score(y_test, y_test_pred)

    print("=== Desempenho no Test Set ===")
    print(f"RMSE: {rmse_test:.4f}")
    print(f"R²:   {r2_test:.4f}")

    # 4) Veja as previsões para validação (submissão)
    print("Previsões para Validation Set (67 amostras):")
    print(y_val_pred)

