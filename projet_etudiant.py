# %% [markdown]
# # PROJET : Analyse de données · régression linéaire, ANOVA, khi-2
# Nom, prénom : ......................   Groupe : ............
# Bibliothèques : pandas, numpy, scipy, matplotlib, seaborn (statsmodels autorisé en option).
# Complétez les zones "# TODO". Commentez vos choix et interprétez TOUS vos résultats par écrit.
# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression

sns.set_theme(style="whitegrid")
ALPHA = 0.05

# %% PARTIE 0 : chargement et nettoyage (2 pts)
df = pd.read_csv("donnees_projet.csv")

# TODO 0.1 : dimensions, types, valeurs manquantes par variable

print("\n DIMENSIONS :")
print(df.shape)
print("\n TYPES :")
print(df.dtypes)
print("\n VALEURS MANQUANTES :")
print(df.isna().sum())

# TODO 0.2 : repérer une valeur aberrante dans heures_etude (describe, boxplot). Que faites-vous ? Justifiez.

print("\n STATISTIQUES DE HEURES_ETUDE :")
heures_etude_stat = df["heures_etude"].describe()
print(heures_etude_stat)
q1 = heures_etude_stat["25%"]
q2 = heures_etude_stat["50%"]
median = q2
q3 = heures_etude_stat["75%"]
iqr = q3 - q1
limite_sup = q3 + 1.5 * iqr
print("Limite superieure :", limite_sup)

valeurs_aberrantes = df[(df["heures_etude"] > limite_sup)]
print("\n VALEURS ABERRANTES :")
print(valeurs_aberrantes[["id", "heures_etude"]])


plt.figure(figsize=(10, 4))
sns.boxplot(x=df["heures_etude"])
plt.title("boite à moustache des heures d'étude")
plt.xlabel("Heures d'étude par semaine")
plt.show()

df["heures_etude_corrige"] = df["heures_etude"]
df.loc[df["heures_etude_corrige"] == 95, "heures_etude_corrige"] = np.nan
print(df["heures_etude_corrige"].describe())

# JUSTIFICATION :
# La methode IQR a detecte 6 valeurs depassant la limite superieure.
# Les 5 valeurs elevees sont conservees car elles restent plausibles.
# En revanche, 95 heures par semaine est une valeur extremement elevee et peu realiste.
# Nous la remplacons par NaN pour eviter de fausser les analyses, tout en conservant les donnees originales.

# TODO 0.3 : construire le tableau d'analyse "d" sans valeurs manquantes sur les variables utilisées

d = df.drop(columns=["id", "heures_etude"]).copy().dropna()

# %% PARTIE 1 : analyse descriptive (2 pts)
# TODO 1.1 : moyenne, écart-type, min, max de moyenne_bac, heures_etude, note_stat

print(d[["moyenne_bac", "heures_etude_corrige", "note_stat"]].describe())

# TODO 1.2 : moyenne de note_stat par filière ; effectifs de chaque modalité qualitative

print("\n moyenne de note_stat par filière")
print(d.groupby("filiere")["note_stat"].mean())
print("\n EFFECTIFS PAR FILIERE :")
print(d["filiere"].value_counts())

print("\n EFFECTIFS PAR SEXE :")
print(d["sexe"].value_counts())

print("\n EFFECTIFS PAR LOGEMENT :")
print(d["logement"].value_counts())

print("\n EFFECTIFS PAR ASSIDUITE :")
print(d["assiduite"].value_counts())

print("\n EFFECTIFS PAR REUSSITE :")
print(d["reussite"].value_counts())
# TODO 1.3 : 2 graphiques commentés (histogramme et boxplot par exemple)

plt.figure(figsize=(8, 5))

plt.hist(d["note_stat"], bins=10, edgecolor="black")

plt.title("Distribution des notes en statistique")
plt.xlabel("Note en statistique")
plt.ylabel("Nombre d'étudiants")

plt.show()

plt.figure(figsize=(8, 5))

sns.boxplot(data=d, x="filiere", y="note_stat")

plt.title("Distribution des notes par filière")
plt.xlabel("Filière")
plt.ylabel("Note en statistique")

plt.show()

# %% PARTIE 2 : régression linéaire simple, note_stat ~ heures_etude (6 pts)
# TODO 2.1 : nuage de points. La relation semble-t-elle linéaire ?

plt.figure(figsize=(8, 5))

plt.scatter(d["heures_etude_corrige"], d["note_stat"])

plt.title("Relation entre les heures d'étude et la note en statistique")
plt.xlabel("Heures d'étude par semaine")
plt.ylabel("Note en statistique")

plt.show()

# Le graphique montre une tendance positive, les étudiants qui étudient davantage ont généralement de meilleures notes.

# TODO 2.2 : coefficient de corrélation r et son test

correlation = stats.pearsonr(d["heures_etude_corrige"], d["note_stat"])
r = correlation.statistic
p = correlation.pvalue

print("\n CORRELATION DE PEARSON :")
print("Coefficient r :", r)
print("P-value :", p)

# Le coefficient r est 0.680 ce qui indique une corrélation positive assez forte entre les heures d'étude et les notes en statistique.
# La p-value est inférieure à 0.05, ce qui confirme que cette relation est statistiquement significative.

# TODO 2.3 : estimer la droite (stats.linregress) ; écrire l'équation ; interpréter pente et ordonnée

regression = stats.linregress(d["heures_etude_corrige"], d["note_stat"])
pente = regression.slope
ordonnee = regression.intercept

print("\n DROITE DE REGRESSION :")
print("Pente :", pente)
print("Ordonnee a l'origine :", ordonnee)

print(
    f"Equation : note_stat = {pente.round(3)} * heures_etude_corrige + {ordonnee.round(3)}"
)

# Chaque heure supplémentaire est associée à une hausse moyenne d'environ 0.55 point.
# Pour 0 heure d'étude, le modèle prédit une note de 6.15/20.


# TODO 2.4 : R², p-value de la pente, intervalle de confiance de la pente

r2 = regression.rvalue**2
p_pente = regression.pvalue


print("R² :", r2.round(2), " ou ", r2.round(2) * 100, "%")


print("\n TEST DE SIGNIFICATIVITE DE LA PENTE :")
print("P-value :", p_pente)

t_critique = stats.t.ppf(1 - 0.05 / 2, len(d) - 2)

borne_inf = pente - t_critique * regression.stderr
borne_sup = pente + t_critique * regression.stderr

print("\n INTERVALLE DE CONFIANCE A 95 % :")
print("Borne inferieure :", borne_inf.round(3))
print("Borne superieure :", borne_sup.round(3))

# TODO 2.5 : retrouver pente et ordonnée "à la main" (formules du cours) et comparer

moyenne_h = d["heures_etude_corrige"].mean()
moyenne_n = d["note_stat"].mean()

numi = 0
deno = 0

for x_i, y_i in d[["heures_etude_corrige", "note_stat"]].itertuples(index=False):
    numi += (x_i - moyenne_h) * (y_i - moyenne_n)
    deno += (x_i - moyenne_h) ** 2

pente_m = numi / deno

print("Pente calculée manuellement :", pente_m)
print("Pente avec scipy :", pente)


ordonnee_m = moyenne_n - pente_m * moyenne_h

print("Ordonnee calculée manuellement :", ordonnee_m)
print("Ordonnee avec scipy :", ordonnee)

# Les calculs manuels donnent les mêmes résultats que stats.linregress().

# TODO 2.6 : prédire la note pour 10 h d'étude par semaine ; peut-on prédire pour 40 h ? pourquoi ?


def note_par_h(h):
    return pente * h + ordonnee


print("\n note pour 10h par semaine : ", note_par_h(10).round(2))
print("\n note pour 40h par semaine : ", note_par_h(40).round(2))


# Pour 10 heures d'étude, le modèle prédit une note de 11.63/20.
# Pour 40 heures, le modèle prédit 28.08/20, cette prédiction n'est pas fiable car 40 heures dépasse la plage observée.

# TODO 2.7 : résidus : graphique résidus vs ajustées, QQ-plot, test de Shapiro ; conditions respectées ?

residus = []

for n, h in d[["note_stat", "heures_etude_corrige"]].itertuples(index=False):
    residus.append(n - note_par_h(h))

valeurs_ajustees = []

for h in d["heures_etude_corrige"]:
    valeurs_ajustees.append(note_par_h(h))

plt.figure(figsize=(8, 5))
plt.scatter(valeurs_ajustees, residus)
plt.axhline(y=0, color="red", linestyle="--")

plt.title("Résidus en fonction des valeurs ajustées")
plt.xlabel("Valeurs ajustées")
plt.ylabel("Résidus")

plt.show()

plt.figure(figsize=(8, 5))

stats.probplot(residus, dist="norm", plot=plt)

plt.title("QQ-plot des résidus")
plt.xlabel("Quantiles théoriques")
plt.ylabel("Quantiles observés")

plt.show()

shapiro = stats.shapiro(residus)

print("\n TEST DE SHAPIRO-WILK :")
print("Statistique W :", shapiro.statistic)
print("P-value :", shapiro.pvalue)


# Les résidus sont répartis autour de zéro sans tendance particulière.
# Leur dispersion reste assez régulière sans effet d'entonnoir.
# Les points du QQ-plot suivent globalement la droite.
# La p-value de Shapiro est supérieure à 0.05.
# Les conditions de la régression semblent donc globalement respectées.

# BONUS 2.8 : régression multiple (heures_etude + moyenne_bac) avec scikit-learn ou statsmodels ; comparer les R²

# BONUS 2.8 : régression linéaire multiple

x = d[["heures_etude_corrige", "moyenne_bac"]]
y = d["note_stat"]

modele_multiple = LinearRegression()
modele_multiple.fit(x, y)

r2_multiple = modele_multiple.score(x, y)

print("\n REGRESSION MULTIPLE :")
print("Coefficient heures_etude :", modele_multiple.coef_[0])
print("Coefficient moyenne_bac :", modele_multiple.coef_[1])
print("Ordonnée à l'origine :", modele_multiple.intercept_)
print("R² multiple :", round(r2_multiple, 2))
print("\n COMPARAISON :")
print("R² :", r2.round(2))
print("R² multiple :", round(r2_multiple, 2))

# L'ajout de la moyenne du bac augmente le R² de 46.2 % à 51.3 %.
# Le modèle multiple explique donc mieux les notes.

# %% PARTIE 3 : ANOVA à un facteur, note_stat ~ filiere (5 pts)
# TODO 3.1 : H0 et H1 ; boxplot de la note par filière

# h0 : les moyennes des trois filieres sont egales.
# h1 : au moins une filiere a une moyenne differente.

plt.figure(figsize=(8, 5))
sns.boxplot(data=d, x="filiere", y="note_stat")
plt.title("Notes en statistique selon la filière")
plt.xlabel("Filière")
plt.ylabel("Note en statistique")
plt.show()

# Les notes semblent plus élevées en Informatique et plus faibles en Chimie.

# TODO 3.2 : vérifier les conditions : normalité par groupe (Shapiro), égalité des variances (Levene)
#
# # TEST DE NORMALITÉ DE SHAPIRO PAR FILIÈRE
for f, g in d.groupby("filiere"):
    res = stats.shapiro(g["note_stat"])
    print(f"\n{f} :")
    print("W :", res.statistic)
    print("P-value :", res.pvalue)

# TEST D'ÉGALITÉ DES VARIANCES DE LEVENE
groupes = [groupe["note_stat"].values for _, groupe in d.groupby("filiere")]

levene = stats.levene(*groupes)

print("\nTEST DE LEVENE :")
print("Statistique :", levene.statistic)
print("P-value :", levene.pvalue)

# La normalité et l'égalité des variances sont respectées.
# Les conditions de l'ANOVA sont donc satisfaites.

# TODO 3.3 : calculer SC_entre, SC_intra, CM, F, ddl à la main puis comparer avec stats.f_oneway

SC_entre = 0
for f, g in d.groupby("filiere"):
    SC_entre += len(g) * (g["note_stat"].mean() - d["note_stat"].mean()) ** 2

SC_intra = 0
for f, g in d.groupby("filiere"):
    for n in g["note_stat"]:
        SC_intra += (n - g["note_stat"].mean()) ** 2

ddl_entre = d["filiere"].nunique() - 1
ddl_intra = len(d) - d["filiere"].nunique()

CM_entre = SC_entre / ddl_entre
CM_intra = SC_intra / ddl_intra

F_manuel = CM_entre / CM_intra

print("\nANOVA MANUELLE :")
print("SC entre :", SC_entre)
print("SC intra :", SC_intra)
print("CM entre :", CM_entre)
print("CM intra :", CM_intra)
print("DDL entre :", ddl_entre)
print("DDL intra :", ddl_intra)
print("F manuel :", F_manuel)

anova = stats.f_oneway(*groupes)
print("\nANOVA AVEC SCIPY :")
print("F :", anova.statistic)
print("P-value :", anova.pvalue)

# les calculs manuels correspondent aux resultat de scipy.

# TODO 3.4 : F critique (stats.f.ppf) ; décision par comparaison F / F critique ET par la p-value

F_critique = stats.f.ppf(1 - 0.05, ddl_entre, ddl_intra)

print("\nComparaison F avec F critique :")
print("\nF :", F_manuel)
print("\nF critique :", F_critique)

print("\nP value : ", anova.pvalue)

# F est supérieur à F critique et la p-value est inférieure à 0.05 alors on rejette h0

# TODO 3.5 : taille d'effet η² ; interprétation

n2 = SC_entre / (SC_entre + SC_intra)
print("Eta² :", n2, " ou ", round(n2 * 100, 2), "%")

# TODO 3.6 : comparaisons par paires (stats.tukey_hsd) : quelles filières diffèrent ?

t = stats.tukey_hsd(*groupes)
print("\nTEST DE TUKEY :")
print("Statistiques : ", t.statistic)
print("P-values : ", t.pvalue)

# Le test de tukey montre des différences significatives entre les trois filières.
# Toutes les comparaisons ont une p-value inférieure à 0.05.

# BONUS 3.7 : test non paramétrique (stats.kruskal) ; conclusion identique ?

# %% PARTIE 4 : khi-2 d'indépendance (5 pts)
# Deux couples à tester : (logement, assiduite) et (sexe, filiere)
# Pour CHAQUE couple :
# TODO 4.1 : H0, H1 ; tableau croisé observé (pd.crosstab)
# TODO 4.2 : tableau des effectifs théoriques calculés avec la formule du cours (ligne x colonne / total)
# TODO 4.3 : contribution de chaque cellule (O-E)²/E, puis le χ² total
# TODO 4.4 : ddl, χ² critique, p-value ; comparer avec stats.chi2_contingency
# TODO 4.5 : vérifier la condition E >= 5 ; décision ; conclusion en une phrase
# TODO 4.6 : V de Cramér ; graphique en barres empilées en pourcentages
# Question de réflexion : pourquoi un khi-2 significatif ne prouve-t-il pas une relation de cause à effet ?

# %% PARTIE 5 : synthèse (clarté, 0 pt ici, évalué dans le rapport)
# Rédigez dans le rapport une conclusion de 10 lignes reliant les trois méthodes.
