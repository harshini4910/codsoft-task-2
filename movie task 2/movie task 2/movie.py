import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')
print("✅ Setup complete!")
np.random.seed(42)
data = {
    'Director': np.random.choice(['Nolan', 'Tarantino', 'Scorsese', 'Spielberg'], 500),
    'Lead_Actor': np.random.choice(['DiCaprio', 'Pitt', 'DeNiro', 'Hanks'], 500),
    'Genre': np.random.choice(['Drama', 'Action', 'Sci-Fi', 'Comedy'], 500),
    'Budget_M': np.random.lognormal(7, 1, 500).clip(1, 500),
    'Runtime_min': np.random.normal(120, 25, 500).clip(60, 240),
    'Year': np.random.choice(range(2010, 2026), 500)}
data['Rating'] = (6.5 + 
                 pd.Series(data['Director']).map({'Nolan':0.8, 'Scorsese':0.6}).fillna(0) +
                 pd.Series(data['Genre']).map({'Drama':0.3, 'Sci-Fi':0.2}).fillna(0) +
                 np.random.normal(0, 0.8, 500)).clip(1, 10)
df = pd.DataFrame(data)
print(f"Shape: {df.shape}")
print(df.head(3))
plt.figure(figsize=(10, 6))
df['Rating'].hist(bins=30, edgecolor='black', alpha=0.7, color='skyblue')
plt.title('Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.show()
print("✅ Rating distribution plotted!")
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Genre', y='Rating', palette='Set2')
plt.title('Rating by Genre')
plt.xticks(rotation=45)
plt.ylabel('Rating')
plt.tight_layout()
plt.show()
print("✅ Genre analysis complete!")
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Director', y='Rating', palette='Set3')
plt.title('Rating by Director')
plt.xticks(rotation=45)
plt.ylabel('Rating')
plt.tight_layout()
plt.show()
print("✅ Director analysis complete!")
print("📊 Rating Statistics:")
print(df['Rating'].describe())
print("\nTop Directors by Average Rating:")
print(df.groupby('Director')['Rating'].mean().sort_values(ascending=False))
print("\nTop Genres by Average Rating:")
print(df.groupby('Genre')['Rating'].mean().sort_values(ascending=False))

