import { useState, useRef, useEffect } from "react";

const API_URL = "https://api.anthropic.com/v1/messages";

async function callClaude(systemPrompt, userMessage) {
  const res = await fetch(API_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "claude-sonnet-4-20250514",
      max_tokens: 1000,
      system: systemPrompt,
      messages: [{ role: "user", content: userMessage }],
    }),
  });
  const data = await res.json();
  return data.content?.map((c) => c.text || "").join("") || "";
}

// ── colour tokens ──────────────────────────────────────────────
const C = {
  bg: "#0b0f1a",
  panel: "#111827",
  border: "#1e2d45",
  accent: "#00d4aa",
  accentDim: "#00a882",
  gold: "#f5c542",
  rose: "#f43f5e",
  text: "#e2e8f0",
  muted: "#64748b",
  code: "#0f1f2e",
};

// ── slides data ────────────────────────────────────────────────
const SLIDES = [
  {
    id: "welcome",
    title: "ML Mastery: From Zero to Hired",
    subtitle: "Your interactive journey into Machine Learning",
    type: "cover",
    content: null,
  },
  {
    id: "what-is-ml",
    title: "What is Machine Learning?",
    type: "concept",
    employerNote:
      "Employers expect you to explain ML in plain English without jargon — practice the 'explain it to your grandmother' test.",
    content: {
      explanation: `Machine Learning is a branch of Artificial Intelligence where computers learn patterns from data **without being explicitly programmed** for every rule.\n\nInstead of writing rules like "if temperature > 30 → it's hot", you show the model thousands of examples and it figures out the rule itself.\n\n**Three main types:**\n- **Supervised Learning** — learns from labelled examples (e.g., predicting house prices)\n- **Unsupervised Learning** — finds hidden patterns with no labels (e.g., customer segmentation)\n- **Reinforcement Learning** — learns by trial and reward (e.g., game-playing agents)`,
      analogy:
        "Think of ML like teaching a child. You don't explain the rules of recognising a cat — you show thousands of cat pictures until the child 'gets it'.",
    },
  },
  {
    id: "setup",
    title: "Setting Up Your Workspace",
    type: "setup",
    employerNote:
      "Being able to set up a reproducible environment from scratch is a core hiring signal. Employers often ask: 'walk me through how you'd start a new ML project.'",
    content: {
      steps: [
        {
          label: "1. Google Colab",
          detail:
            "Go to colab.research.google.com → New Notebook. Free GPU/TPU available under Runtime → Change Runtime Type.",
        },
        {
          label: "2. Kaggle Dataset",
          detail:
            "Sign up at kaggle.com → Go to a dataset → Click the three-dot menu → Copy API command → Paste in Colab.",
        },
        {
          label: "3. Kaggle API in Colab",
          detail:
            "Upload your kaggle.json key, then run: `!mkdir ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json`",
        },
        {
          label: "4. Download Dataset",
          detail:
            "`!kaggle datasets download -d <dataset-path> --unzip` — then list files with `!ls`",
        },
      ],
    },
  },
  {
    id: "import",
    title: "Step 1 — Importing Libraries",
    type: "code",
    employerNote:
      "Always import only what you need — shows discipline. Employers scan imports to understand your tool fluency.",
    testPrompt:
      "Write the standard imports for a machine learning project using pandas, numpy, matplotlib, seaborn, and scikit-learn. Include a comment block at the top.",
    rubric:
      "Should include: pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns, and at least one sklearn import. Comments are a bonus.",
    code: `# ── Standard ML Imports ───────────────────────────────────────
import pandas as pd          # data manipulation
import numpy as np           # numerical computing
import matplotlib.pyplot as plt  # plotting
import seaborn as sns        # statistical visualisation

# Scikit-learn modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Display settings
pd.set_option('display.max_columns', None)
%matplotlib inline
print("All libraries imported successfully ✅")`,
  },
  {
    id: "loading",
    title: "Step 2 — Loading the Dataset",
    type: "code",
    employerNote:
      "Always do an initial exploration right after loading — df.head(), df.info(), df.describe(). Skipping this is a red flag in interviews.",
    testPrompt:
      "Write code to load a CSV file called 'data.csv' into a pandas DataFrame called df, then display the first 5 rows and show the shape of the dataset.",
    rubric:
      "Must use pd.read_csv(), assign to df, call df.head(), and print or display df.shape.",
    code: `# Load dataset (after Kaggle download)
df = pd.read_csv('your_dataset.csv')

# ── Exploratory peek ──────────────────────────────────────────
print(f"Dataset shape: {df.shape}")   # (rows, columns)
print("\\nFirst 5 rows:")
display(df.head())

print("\\nColumn types & non-null counts:")
df.info()

print("\\nBasic statistics:")
display(df.describe())`,
  },
  {
    id: "eda",
    title: "Step 3 — Exploratory Data Analysis (EDA)",
    type: "code",
    employerNote:
      "EDA is where most of the real insight comes from. Employers love candidates who ask 'what story does the data tell?' before modelling.",
    testPrompt:
      "Write code to check for missing values in a DataFrame df, show the percentage of missing values per column, and plot a heatmap of the missing data using seaborn.",
    rubric:
      "Should use df.isnull().sum(), calculate percentage, and call sns.heatmap() on df.isnull().",
    code: `# ── Missing Values ────────────────────────────────────────────
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({'Missing': missing, 'Percent': missing_pct})
print(missing_df[missing_df['Missing'] > 0].sort_values('Percent', ascending=False))

# Heatmap of missing data
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title('Missing Data Heatmap')
plt.tight_layout()
plt.show()

# ── Distribution of numeric columns ──────────────────────────
df.hist(figsize=(14, 10), bins=30, color='#00d4aa', edgecolor='black')
plt.suptitle('Feature Distributions', fontsize=16)
plt.tight_layout()
plt.show()

# ── Correlation heatmap ───────────────────────────────────────
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt='.2f',
            cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()`,
  },
  {
    id: "cleaning",
    title: "Step 4 — Data Cleaning",
    type: "code",
    employerNote:
      "Up to 80% of a data scientist's time is cleaning data. Showing systematic cleaning with reasoning for each decision is a top interview differentiator.",
    testPrompt:
      "Write code to handle missing values in a DataFrame df: fill numeric columns with the median and categorical columns with the mode. Then drop any duplicate rows.",
    rubric:
      "Should use df.select_dtypes(), fillna() with median() and mode()[0], and df.drop_duplicates().",
    code: `# ── Handle Missing Values ────────────────────────────────────
# Numeric columns → fill with median (robust to outliers)
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Categorical columns → fill with mode
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# ── Remove Duplicates ─────────────────────────────────────────
before = len(df)
df = df.drop_duplicates()
print(f"Removed {before - len(df)} duplicate rows")

# ── Remove Outliers (IQR Method) ──────────────────────────────
def remove_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]

# Apply to each numeric column
for col in numeric_cols:
    df = remove_outliers_iqr(df, col)

print(f"Clean dataset shape: {df.shape}")
print("Missing values remaining:", df.isnull().sum().sum())`,
  },
  {
    id: "encoding",
    title: "Step 5 — Feature Engineering & Encoding",
    type: "code",
    employerNote:
      "Feature engineering is often the difference between a mediocre and a great model. Interviewers frequently ask: 'What features did you create and why?'",
    testPrompt:
      "Write code to one-hot encode all categorical columns in a DataFrame df using pandas get_dummies, then scale all numeric features using StandardScaler from sklearn.",
    rubric:
      "Should use pd.get_dummies() with drop_first=True, instantiate StandardScaler, call fit_transform on numeric columns.",
    code: `# ── One-Hot Encoding ─────────────────────────────────────────
df_encoded = pd.get_dummies(df, drop_first=True)  # drop first avoids multicollinearity
print(f"Shape after encoding: {df_encoded.shape}")

# ── Feature Scaling ───────────────────────────────────────────
# Separate target (change 'target_column' to your actual target)
TARGET = 'target_column'
X = df_encoded.drop(TARGET, axis=1)
y = df_encoded[TARGET]

# Scale features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

print("\\nScaled features (mean ≈ 0, std ≈ 1):")
print(X_scaled.describe().loc[['mean', 'std']].round(2))`,
  },
  {
    id: "splitting",
    title: "Step 6 — Train / Test Split",
    type: "code",
    employerNote:
      "Never train and test on the same data — this is a fundamental concept. Interviewers will probe on this: 'Why 80/20?', 'What is data leakage?'",
    testPrompt:
      "Write code to split a dataset with features X_scaled and target y into 80% training and 20% test sets with a random_state of 42. Print the sizes of each set.",
    rubric:
      "Must use train_test_split with test_size=0.2, random_state=42, and print shapes of X_train, X_test, y_train, y_test.",
    code: `# ── Train / Test Split ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,      # 20% for testing
    random_state=42     # reproducibility
)

print(f"Training set:  X={X_train.shape}  y={y_train.shape}")
print(f"Test set:      X={X_test.shape}   y={y_test.shape}")
print(f"\\nClass balance in training set:")
print(y_train.value_counts(normalize=True).round(3))  # for classification`,
  },
  {
    id: "regression",
    title: "Step 7 — Linear Regression Model",
    type: "code",
    employerNote:
      "Always start simple. Interviewers want to see you establish a baseline before jumping to complex models. Linear regression IS a valid final answer if it performs well.",
    testPrompt:
      "Write code to train a LinearRegression model on X_train and y_train, make predictions on X_test, and calculate MSE, RMSE, and R² score.",
    rubric:
      "Should instantiate LinearRegression(), call fit(), predict(), then compute mean_squared_error and r2_score. Bonus: np.sqrt(MSE) for RMSE.",
    code: `# ── Train Model ───────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)

# ── Predict ───────────────────────────────────────────────────
y_pred = model.predict(X_test)

# ── Evaluate ──────────────────────────────────────────────────
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print(f"📊 Model Performance")
print(f"   MSE  : {mse:.4f}")
print(f"   RMSE : {rmse:.4f}  ← same units as target")
print(f"   R²   : {r2:.4f}   ← 1.0 is perfect")

# ── Visualise: Actual vs Predicted ───────────────────────────
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='#00d4aa')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect fit')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted')
plt.legend()
plt.tight_layout()
plt.show()

# ── Coefficients ──────────────────────────────────────────────
coef_df = pd.DataFrame({'Feature': X_train.columns,
                         'Coefficient': model.coef_})
coef_df = coef_df.sort_values('Coefficient', ascending=False)
print("\\nTop feature coefficients:")
print(coef_df.head(10))`,
  },
  {
    id: "metrics",
    title: "Key Metrics Employers Look For",
    type: "metrics",
    content: {
      technical: [
        {
          skill: "Python Proficiency",
          detail:
            "Clean, commented, PEP-8 code. No spaghetti logic. Functions over repeated code.",
        },
        {
          skill: "Data Wrangling",
          detail:
            "Pandas, NumPy, handling nulls, outliers, encoding — fast and methodical.",
        },
        {
          skill: "Statistical Foundation",
          detail:
            "Understand p-values, distributions, correlation ≠ causation, hypothesis testing.",
        },
        {
          skill: "Model Selection",
          detail:
            "Know when to use regression vs. classification vs. clustering. Justify choices.",
        },
        {
          skill: "Evaluation Metrics",
          detail:
            "R², RMSE, MAE for regression; accuracy, precision, recall, F1, AUC for classification.",
        },
        {
          skill: "Reproducibility",
          detail:
            "random_state=42, requirements.txt, clear notebook structure, version control.",
        },
        {
          skill: "Visualisation",
          detail:
            "Clear, labelled plots that tell a story. matplotlib + seaborn minimum.",
        },
        {
          skill: "Version Control",
          detail:
            "GitHub with meaningful commit messages. Employers WILL check your repos.",
        },
      ],
      soft: [
        {
          skill: "Communication",
          detail:
            "Explain findings to non-technical stakeholders. 'So what?' thinking.",
        },
        {
          skill: "Problem Framing",
          detail:
            "Before coding, define the problem, success metric, and constraints.",
        },
        {
          skill: "Curiosity",
          detail: "Ask why, probe anomalies, never accept results blindly.",
        },
        {
          skill: "Business Acumen",
          detail:
            "Translate model output to business value. ROI thinking.",
        },
      ],
      interview: [
        "Walk me through your ML workflow from scratch",
        "What is overfitting and how do you handle it?",
        "Explain bias-variance tradeoff",
        "When would you use Random Forest over Linear Regression?",
        "How do you handle imbalanced datasets?",
        "What is cross-validation and why is it important?",
        "Explain how gradient descent works",
        "How would you deploy this model to production?",
      ],
    },
  },
  {
    id: "projects",
    title: "Portfolio Projects — UN SDG Aligned",
    type: "projects",
    content: null,
  },
  {
    id: "review",
    title: "Employer Notebook Review",
    type: "review",
    content: null,
  },
];

const SDG_PROJECTS = [
  {
    id: "maternal",
    sdg: "SDG 3 — Good Health",
    icon: "🏥",
    title: "Maternal Mortality Risk Predictor",
    tagline: "Predict high-risk pregnancies to save lives",
    dataset: "WHO Maternal Health Risk Dataset (Kaggle)",
    skills: ["Classification", "Logistic Regression", "Random Forest", "SMOTE for imbalance"],
    impact: "Directly addresses preventable maternal deaths — one of the UN's top priorities",
    github:
      "Structure: data cleaning → EDA → risk stratification → model comparison → explainability (SHAP)",
    walkthrough: [
      "Download 'Maternal Health Risk Data Set' from Kaggle",
      "EDA: age distribution, BP vs risk level, blood sugar patterns",
      "Handle class imbalance with SMOTE or class_weight='balanced'",
      "Train Logistic Regression baseline, then Random Forest",
      "Use SHAP values to explain which features drive risk",
      "Build a simple Gradio interface for prediction",
      "Write a README linking to UN SDG 3 targets",
    ],
  },
  {
    id: "girls-education",
    sdg: "SDG 4 & 5 — Education + Gender Equality",
    icon: "📚",
    title: "Girls' School Dropout Prediction",
    tagline: "Identify at-risk students before they leave school",
    dataset: "Student Performance Dataset / UNESCO EdStats (Kaggle)",
    skills: ["Classification", "Feature Importance", "Decision Trees", "Policy recommendations"],
    impact: "130M girls are out of school globally — ML can target interventions",
    github:
      "Structure: country-level EDA → gender gap analysis → dropout predictor → policy brief",
    walkthrough: [
      "Load UNESCO or student performance dataset from Kaggle",
      "Create 'dropout risk' target from attendance + grades features",
      "EDA: compare gender gaps across regions/schools",
      "Train Decision Tree (interpretable for policy makers)",
      "Feature importance: what predicts dropout most?",
      "Visualise regional heatmaps with folium",
      "Write policy recommendations section in the notebook",
    ],
  },
  {
    id: "wage-gap",
    sdg: "SDG 5 — Gender Equality",
    icon: "⚖️",
    title: "Gender Wage Gap Analyser",
    tagline: "Quantify and explain the gender pay gap with data",
    dataset: "US Census / LinkedIn Salary Data (Kaggle)",
    skills: ["Regression", "Statistical testing", "Fairness ML", "Data storytelling"],
    impact: "Women earn 77 cents for every dollar men earn — make it measurable",
    github:
      "Structure: salary EDA → gap decomposition → regression → fairness audit → visualisation",
    walkthrough: [
      "Load salary/census dataset from Kaggle",
      "EDA: median salary by gender, industry, education level",
      "Build regression controlling for experience, role, education",
      "Residual = unexplained gap after controlling for job factors",
      "Statistical significance testing (t-test, confidence intervals)",
      "Interactive seaborn/plotly charts showing gap by sector",
      "Conclude with data-driven policy recommendations",
    ],
  },
  {
    id: "disease",
    sdg: "SDG 3 — Good Health",
    icon: "🧬",
    title: "Diabetes Early Detection System",
    tagline: "Predict diabetes onset from routine health metrics",
    dataset: "Pima Indians Diabetes Dataset (Kaggle)",
    skills: ["Binary Classification", "ROC-AUC", "Hyperparameter Tuning", "Medical ML ethics"],
    impact: "537M people live with diabetes — early detection saves lives and costs",
    github:
      "Structure: medical EDA → model comparison → threshold tuning → explainability → ethical discussion",
    walkthrough: [
      "Load Pima Indians Diabetes dataset",
      "EDA: glucose, BMI, age distributions by outcome",
      "Compare Logistic Regression, Random Forest, XGBoost",
      "Plot ROC curves, choose threshold based on recall (sensitivity)",
      "SHAP explainability: which features matter most?",
      "Discuss model ethics: false negatives in medical AI",
      "Create a prediction function with confidence scores",
    ],
  },
];

// ── Component helpers ──────────────────────────────────────────

function Badge({ children, color = C.accent }) {
  return (
    <span
      style={{
        background: color + "22",
        color,
        border: `1px solid ${color}44`,
        borderRadius: 4,
        padding: "2px 10px",
        fontSize: 11,
        fontWeight: 700,
        letterSpacing: 1,
        textTransform: "uppercase",
      }}
    >
      {children}
    </span>
  );
}

function CodeBlock({ code }) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  return (
    <div style={{ position: "relative", marginTop: 12 }}>
      <button
        onClick={copy}
        style={{
          position: "absolute",
          top: 8,
          right: 10,
          background: copied ? C.accent : "#1e2d45",
          color: copied ? "#000" : C.muted,
          border: "none",
          borderRadius: 4,
          padding: "3px 10px",
          fontSize: 11,
          cursor: "pointer",
          fontFamily: "monospace",
          zIndex: 2,
        }}
      >
        {copied ? "Copied!" : "Copy"}
      </button>
      <pre
        style={{
          background: C.code,
          border: `1px solid ${C.border}`,
          borderRadius: 8,
          padding: "20px 16px",
          overflowX: "auto",
          fontSize: 13,
          lineHeight: 1.7,
          color: "#a8d8ea",
          margin: 0,
          fontFamily: "'Fira Code', 'Cascadia Code', monospace",
          whiteSpace: "pre",
        }}
      >
        <code>{code}</code>
      </pre>
    </div>
  );
}

function EmployerNote({ note }) {
  return (
    <div
      style={{
        background: "#1a2a1a",
        border: `1px solid ${C.accent}44`,
        borderLeft: `3px solid ${C.accent}`,
        borderRadius: 6,
        padding: "10px 16px",
        marginTop: 16,
        display: "flex",
        gap: 10,
        alignItems: "flex-start",
      }}
    >
      <span style={{ fontSize: 18 }}>💼</span>
      <div>
        <div
          style={{
            color: C.accent,
            fontWeight: 700,
            fontSize: 11,
            letterSpacing: 1,
            textTransform: "uppercase",
            marginBottom: 4,
          }}
        >
          Employer Insight
        </div>
        <div style={{ color: "#94a3b8", fontSize: 13, lineHeight: 1.6 }}>{note}</div>
      </div>
    </div>
  );
}

function CodeTestPanel({ slide }) {
  const [userCode, setUserCode] = useState("");
  const [feedback, setFeedback] = useState(null);
  const [loading, setLoading] = useState(false);

  const submitCode = async () => {
    if (!userCode.trim()) return;
    setLoading(true);
    setFeedback(null);
    try {
      const system = `You are a strict but encouraging ML instructor reviewing a student's Python code submission. 
The task was: "${slide.testPrompt}"
Grading rubric: "${slide.rubric}"
Give feedback in this exact JSON format (no markdown, no backticks):
{"grade":"Pass"|"Partial"|"Fail","score":0-100,"praise":"what they did well","issues":["issue1","issue2"],"correction":"corrected code snippet or key fix","tip":"one interview tip related to this skill"}`;
      const raw = await callClaude(system, `Student code:\n\`\`\`python\n${userCode}\n\`\`\``);
      const clean = raw.replace(/```json|```/g, "").trim();
      setFeedback(JSON.parse(clean));
    } catch {
      setFeedback({ grade: "Error", praise: "Could not parse response. Try again.", issues: [], score: 0, tip: "" });
    }
    setLoading(false);
  };

  const gradeColor = feedback
    ? feedback.grade === "Pass"
      ? C.accent
      : feedback.grade === "Partial"
      ? C.gold
      : C.rose
    : C.muted;

  return (
    <div
      style={{
        marginTop: 20,
        background: "#0a1628",
        border: `1px solid ${C.border}`,
        borderRadius: 10,
        overflow: "hidden",
      }}
    >
      <div
        style={{
          background: "#111827",
          padding: "10px 16px",
          borderBottom: `1px solid ${C.border}`,
          display: "flex",
          alignItems: "center",
          gap: 10,
        }}
      >
        <span style={{ fontSize: 16 }}>✍️</span>
        <span style={{ color: C.gold, fontWeight: 700, fontSize: 13 }}>Test Your Knowledge</span>
        <span style={{ color: C.muted, fontSize: 12 }}>— write the code yourself, then submit for AI review</span>
      </div>
      <div style={{ padding: 16 }}>
        <div
          style={{
            background: "#1a2035",
            borderRadius: 6,
            padding: "10px 14px",
            color: "#94a3b8",
            fontSize: 13,
            marginBottom: 10,
            borderLeft: `3px solid ${C.gold}`,
          }}
        >
          <strong style={{ color: C.gold }}>Task:</strong> {slide.testPrompt}
        </div>
        <textarea
          value={userCode}
          onChange={(e) => setUserCode(e.target.value)}
          placeholder="# Write your Python code here..."
          style={{
            width: "100%",
            minHeight: 140,
            background: C.code,
            color: "#a8d8ea",
            border: `1px solid ${C.border}`,
            borderRadius: 6,
            padding: 14,
            fontFamily: "'Fira Code', monospace",
            fontSize: 13,
            lineHeight: 1.6,
            resize: "vertical",
            boxSizing: "border-box",
            outline: "none",
          }}
        />
        <button
          onClick={submitCode}
          disabled={loading || !userCode.trim()}
          style={{
            marginTop: 10,
            background: loading ? C.muted : C.accent,
            color: loading ? "#fff" : "#000",
            border: "none",
            borderRadius: 6,
            padding: "9px 22px",
            fontWeight: 700,
            fontSize: 13,
            cursor: loading ? "not-allowed" : "pointer",
            letterSpacing: 0.5,
          }}
        >
          {loading ? "Reviewing…" : "Submit for Review →"}
        </button>

        {feedback && (
          <div
            style={{
              marginTop: 16,
              background: "#0f1a2e",
              border: `1px solid ${gradeColor}44`,
              borderRadius: 8,
              padding: 16,
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 12 }}>
              <span
                style={{
                  background: gradeColor,
                  color: "#000",
                  fontWeight: 800,
                  fontSize: 13,
                  padding: "3px 12px",
                  borderRadius: 4,
                  letterSpacing: 1,
                }}
              >
                {feedback.grade}
              </span>
              <span style={{ color: gradeColor, fontWeight: 700, fontSize: 15 }}>
                {feedback.score}/100
              </span>
            </div>
            {feedback.praise && (
              <div style={{ color: "#86efac", marginBottom: 8, fontSize: 13 }}>
                ✅ <strong>What you did well:</strong> {feedback.praise}
              </div>
            )}
            {feedback.issues && feedback.issues.length > 0 && (
              <div style={{ marginBottom: 8 }}>
                <div style={{ color: C.rose, fontWeight: 700, fontSize: 12, marginBottom: 4 }}>
                  ⚠ Issues:
                </div>
                {feedback.issues.map((iss, i) => (
                  <div key={i} style={{ color: "#fca5a5", fontSize: 13, marginBottom: 4 }}>
                    • {iss}
                  </div>
                ))}
              </div>
            )}
            {feedback.correction && (
              <div style={{ marginBottom: 8 }}>
                <div style={{ color: C.gold, fontWeight: 700, fontSize: 12, marginBottom: 6 }}>
                  🔧 Key Fix:
                </div>
                <pre
                  style={{
                    background: C.code,
                    borderRadius: 5,
                    padding: "10px 12px",
                    color: "#a8d8ea",
                    fontSize: 12,
                    overflowX: "auto",
                    margin: 0,
                    fontFamily: "monospace",
                  }}
                >
                  {feedback.correction}
                </pre>
              </div>
            )}
            {feedback.tip && (
              <div
                style={{
                  marginTop: 10,
                  background: "#1a2a1a",
                  borderLeft: `3px solid ${C.accent}`,
                  padding: "8px 12px",
                  borderRadius: 4,
                  color: C.accent,
                  fontSize: 12,
                }}
              >
                💼 <strong>Interview tip:</strong> {feedback.tip}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ── PROJECT WALKTHROUGH ────────────────────────────────────────
function ProjectWalkthrough({ project, onBack }) {
  const [step, setStep] = useState(0);
  const [question, setQuestion] = useState("");
  const [qAnswer, setQAnswer] = useState("");
  const [qLoading, setQLoading] = useState(false);

  const askQuestion = async () => {
    if (!question.trim()) return;
    setQLoading(true);
    const sys = `You are an expert ML mentor helping a student build the project: "${project.title}". 
Answer concisely with code examples where relevant. Be encouraging and practical.`;
    const ans = await callClaude(sys, question);
    setQAnswer(ans);
    setQLoading(false);
  };

  return (
    <div style={{ padding: "24px 0" }}>
      <button
        onClick={onBack}
        style={{
          background: "none",
          border: `1px solid ${C.border}`,
          color: C.muted,
          borderRadius: 6,
          padding: "6px 14px",
          cursor: "pointer",
          marginBottom: 20,
          fontSize: 13,
        }}
      >
        ← Back to Projects
      </button>

      <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 6 }}>
        <span style={{ fontSize: 36 }}>{project.icon}</span>
        <div>
          <Badge color={C.accent}>{project.sdg}</Badge>
          <h2 style={{ color: C.text, margin: "4px 0 0", fontSize: 22, fontFamily: "Georgia, serif" }}>
            {project.title}
          </h2>
        </div>
      </div>
      <p style={{ color: C.muted, marginBottom: 20 }}>{project.tagline}</p>

      <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 24 }}>
        {project.skills.map((s) => (
          <Badge key={s} color={C.gold}>{s}</Badge>
        ))}
      </div>

      <div style={{ background: "#0f1a2e", borderRadius: 10, padding: 20, marginBottom: 24 }}>
        <div style={{ color: C.accent, fontWeight: 700, marginBottom: 4, fontSize: 13, letterSpacing: 1 }}>
          📦 DATASET
        </div>
        <div style={{ color: C.text }}>{project.dataset}</div>
      </div>

      <div style={{ color: C.text, fontWeight: 700, marginBottom: 12 }}>
        📋 Step-by-Step Walkthrough
      </div>

      <div style={{ display: "flex", gap: 10, marginBottom: 16, flexWrap: "wrap" }}>
        {project.walkthrough.map((_, i) => (
          <button
            key={i}
            onClick={() => setStep(i)}
            style={{
              background: step === i ? C.accent : "#1e2d45",
              color: step === i ? "#000" : C.muted,
              border: "none",
              borderRadius: 20,
              padding: "4px 14px",
              fontWeight: 700,
              fontSize: 12,
              cursor: "pointer",
            }}
          >
            Step {i + 1}
          </button>
        ))}
      </div>

      <div
        style={{
          background: "#111827",
          border: `1px solid ${C.border}`,
          borderLeft: `4px solid ${C.accent}`,
          borderRadius: 8,
          padding: "16px 20px",
          marginBottom: 20,
        }}
      >
        <div style={{ color: C.accent, fontWeight: 700, fontSize: 12, marginBottom: 6 }}>
          STEP {step + 1} OF {project.walkthrough.length}
        </div>
        <div style={{ color: C.text, fontSize: 15, lineHeight: 1.7 }}>
          {project.walkthrough[step]}
        </div>
        <div style={{ display: "flex", gap: 10, marginTop: 14 }}>
          <button
            onClick={() => setStep(Math.max(0, step - 1))}
            disabled={step === 0}
            style={{
              background: "#1e2d45",
              color: step === 0 ? C.muted : C.text,
              border: "none",
              borderRadius: 6,
              padding: "6px 16px",
              cursor: step === 0 ? "not-allowed" : "pointer",
              fontSize: 13,
            }}
          >
            ← Prev
          </button>
          <button
            onClick={() => setStep(Math.min(project.walkthrough.length - 1, step + 1))}
            disabled={step === project.walkthrough.length - 1}
            style={{
              background: C.accent,
              color: "#000",
              border: "none",
              borderRadius: 6,
              padding: "6px 16px",
              fontWeight: 700,
              cursor: step === project.walkthrough.length - 1 ? "not-allowed" : "pointer",
              fontSize: 13,
            }}
          >
            Next →
          </button>
        </div>
      </div>

      {/* Ask mentor */}
      <div style={{ background: "#0a1628", border: `1px solid ${C.border}`, borderRadius: 10, padding: 16 }}>
        <div style={{ color: C.gold, fontWeight: 700, marginBottom: 10, fontSize: 13 }}>
          🎓 Ask Your Mentor Anything About This Project
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && askQuestion()}
            placeholder="e.g. How do I handle class imbalance here?"
            style={{
              flex: 1,
              background: C.panel,
              border: `1px solid ${C.border}`,
              color: C.text,
              borderRadius: 6,
              padding: "8px 12px",
              fontSize: 13,
              outline: "none",
            }}
          />
          <button
            onClick={askQuestion}
            disabled={qLoading}
            style={{
              background: C.gold,
              color: "#000",
              border: "none",
              borderRadius: 6,
              padding: "8px 16px",
              fontWeight: 700,
              cursor: "pointer",
              fontSize: 13,
            }}
          >
            {qLoading ? "…" : "Ask"}
          </button>
        </div>
        {qAnswer && (
          <div
            style={{
              marginTop: 14,
              background: "#111827",
              borderRadius: 8,
              padding: 14,
              color: "#94a3b8",
              fontSize: 13,
              lineHeight: 1.7,
              whiteSpace: "pre-wrap",
            }}
          >
            {qAnswer}
          </div>
        )}
      </div>
    </div>
  );
}

// ── NOTEBOOK REVIEW ────────────────────────────────────────────
function NotebookReview() {
  const [text, setText] = useState("");
  const [review, setReview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [tweaks, setTweaks] = useState(null);
  const [tweaksLoading, setTweaksLoading] = useState(false);

  const runReview = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setReview(null);
    setTweaks(null);
    const sys = `You are a senior ML hiring manager at a top tech company with 15 years of experience. 
You are reviewing a candidate's ML project notebook. Be thorough, realistic, and constructive.
Respond in JSON only (no backticks, no markdown fences):
{
  "hire_decision": "Strong Yes" | "Yes" | "Maybe" | "No",
  "overall_score": 0-100,
  "executive_summary": "2-3 sentence overall impression",
  "strengths": ["strength1", "strength2", "strength3"],
  "weaknesses": ["weakness1", "weakness2", "weakness3"],
  "missing_for_job": ["thing1", "thing2"],
  "github_tip": "one specific github/portfolio tip",
  "interview_prediction": "what questions this notebook would likely prompt in an interview"
}`;
    const raw = await callClaude(sys, `Candidate's notebook content:\n${text}`);
    try {
      const clean = raw.replace(/```json|```/g, "").trim();
      setReview(JSON.parse(clean));
    } catch {
      setReview({ hire_decision: "Error", executive_summary: raw, strengths: [], weaknesses: [], overall_score: 0 });
    }
    setLoading(false);
  };

  const getTweaks = async () => {
    if (!review) return;
    setTweaksLoading(true);
    const sys = `You are an expert ML coach. Based on the review feedback, provide 5 specific, actionable code tweaks and improvements. 
Return JSON only (no backticks):
{"tweaks": [{"title":"tweak title","description":"what to do","code":"code snippet or pseudocode","impact":"why this matters to employers"}]}`;
    const raw = await callClaude(
      sys,
      `Review summary: ${review.executive_summary}\nWeaknesses: ${JSON.stringify(review.weaknesses)}\nMissing: ${JSON.stringify(review.missing_for_job)}`
    );
    try {
      const clean = raw.replace(/```json|```/g, "").trim();
      setTweaks(JSON.parse(clean));
    } catch {
      setTweaks({ tweaks: [{ title: "Error", description: raw, code: "", impact: "" }] });
    }
    setTweaksLoading(false);
  };

  const decisionColor =
    review?.hire_decision === "Strong Yes"
      ? C.accent
      : review?.hire_decision === "Yes"
      ? "#86efac"
      : review?.hire_decision === "Maybe"
      ? C.gold
      : C.rose;

  return (
    <div>
      <div style={{ marginBottom: 16 }}>
        <p style={{ color: C.muted, fontSize: 14, lineHeight: 1.7, marginBottom: 12 }}>
          Paste your Jupyter/Colab notebook content below (or paste the key code cells and markdown). 
          An AI employer persona will review it and give you honest, actionable feedback.
        </p>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste your notebook code, markdown, and outputs here..."
          style={{
            width: "100%",
            minHeight: 200,
            background: C.code,
            border: `1px solid ${C.border}`,
            color: C.text,
            borderRadius: 8,
            padding: 14,
            fontFamily: "'Fira Code', monospace",
            fontSize: 13,
            lineHeight: 1.6,
            resize: "vertical",
            boxSizing: "border-box",
            outline: "none",
          }}
        />
        <button
          onClick={runReview}
          disabled={loading || !text.trim()}
          style={{
            marginTop: 10,
            background: loading ? C.muted : C.rose,
            color: "#fff",
            border: "none",
            borderRadius: 6,
            padding: "10px 24px",
            fontWeight: 700,
            fontSize: 14,
            cursor: loading ? "not-allowed" : "pointer",
            letterSpacing: 0.5,
          }}
        >
          {loading ? "Employer is reviewing…" : "🔍 Submit for Employer Review"}
        </button>
      </div>

      {review && (
        <div>
          <div
            style={{
              background: "#0f1a2e",
              border: `2px solid ${decisionColor}`,
              borderRadius: 12,
              padding: 20,
              marginBottom: 20,
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 12 }}>
              <div
                style={{
                  background: decisionColor,
                  color: "#000",
                  fontWeight: 800,
                  padding: "4px 16px",
                  borderRadius: 6,
                  fontSize: 14,
                  letterSpacing: 0.5,
                }}
              >
                {review.hire_decision}
              </div>
              <div style={{ color: decisionColor, fontWeight: 700, fontSize: 20 }}>
                {review.overall_score}/100
              </div>
            </div>
            <p style={{ color: C.text, lineHeight: 1.7, marginBottom: 14 }}>
              {review.executive_summary}
            </p>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
              <div>
                <div style={{ color: "#86efac", fontWeight: 700, fontSize: 12, marginBottom: 8, letterSpacing: 1 }}>
                  ✅ STRENGTHS
                </div>
                {review.strengths?.map((s, i) => (
                  <div key={i} style={{ color: "#94a3b8", fontSize: 13, marginBottom: 6 }}>
                    • {s}
                  </div>
                ))}
              </div>
              <div>
                <div style={{ color: C.rose, fontWeight: 700, fontSize: 12, marginBottom: 8, letterSpacing: 1 }}>
                  ⚠ WEAKNESSES
                </div>
                {review.weaknesses?.map((w, i) => (
                  <div key={i} style={{ color: "#fca5a5", fontSize: 13, marginBottom: 6 }}>
                    • {w}
                  </div>
                ))}
              </div>
            </div>

            {review.missing_for_job?.length > 0 && (
              <div style={{ marginTop: 14, borderTop: `1px solid ${C.border}`, paddingTop: 14 }}>
                <div style={{ color: C.gold, fontWeight: 700, fontSize: 12, marginBottom: 8 }}>
                  🎯 WHAT'S MISSING FOR THE JOB
                </div>
                {review.missing_for_job.map((m, i) => (
                  <div key={i} style={{ color: "#fde68a", fontSize: 13, marginBottom: 4 }}>
                    → {m}
                  </div>
                ))}
              </div>
            )}

            {review.github_tip && (
              <div
                style={{
                  marginTop: 14,
                  background: "#1a2a1a",
                  borderLeft: `3px solid ${C.accent}`,
                  padding: "8px 14px",
                  borderRadius: 4,
                  color: C.accent,
                  fontSize: 13,
                }}
              >
                📂 <strong>GitHub tip:</strong> {review.github_tip}
              </div>
            )}

            {review.interview_prediction && (
              <div
                style={{
                  marginTop: 10,
                  background: "#1a1a2e",
                  borderLeft: `3px solid ${C.gold}`,
                  padding: "8px 14px",
                  borderRadius: 4,
                  color: "#fde68a",
                  fontSize: 13,
                }}
              >
                🎤 <strong>Likely interview questions:</strong> {review.interview_prediction}
              </div>
            )}
          </div>

          <button
            onClick={getTweaks}
            disabled={tweaksLoading}
            style={{
              background: tweaksLoading ? C.muted : C.accent,
              color: tweaksLoading ? "#fff" : "#000",
              border: "none",
              borderRadius: 6,
              padding: "10px 22px",
              fontWeight: 700,
              fontSize: 13,
              cursor: tweaksLoading ? "not-allowed" : "pointer",
              marginBottom: 20,
            }}
          >
            {tweaksLoading ? "Generating tweaks…" : "🔧 Get Specific Improvement Tweaks →"}
          </button>

          {tweaks?.tweaks && (
            <div>
              <div style={{ color: C.text, fontWeight: 700, marginBottom: 14 }}>
                Suggested Tweaks to Land the Job
              </div>
              {tweaks.tweaks.map((t, i) => (
                <div
                  key={i}
                  style={{
                    background: "#111827",
                    border: `1px solid ${C.border}`,
                    borderRadius: 8,
                    padding: 16,
                    marginBottom: 14,
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
                    <span
                      style={{
                        background: C.accent,
                        color: "#000",
                        fontWeight: 800,
                        fontSize: 11,
                        borderRadius: "50%",
                        width: 22,
                        height: 22,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        flexShrink: 0,
                      }}
                    >
                      {i + 1}
                    </span>
                    <span style={{ color: C.text, fontWeight: 700, fontSize: 14 }}>{t.title}</span>
                  </div>
                  <p style={{ color: "#94a3b8", fontSize: 13, margin: "0 0 8px" }}>{t.description}</p>
                  {t.code && (
                    <pre
                      style={{
                        background: C.code,
                        borderRadius: 5,
                        padding: "10px 12px",
                        color: "#a8d8ea",
                        fontSize: 12,
                        overflowX: "auto",
                        margin: "0 0 8px",
                        fontFamily: "monospace",
                      }}
                    >
                      {t.code}
                    </pre>
                  )}
                  <div style={{ color: C.gold, fontSize: 12 }}>
                    💼 <strong>Why employers care:</strong> {t.impact}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── SLIDE RENDERER ─────────────────────────────────────────────
function SlideContent({ slide, setActiveProject }) {
  if (slide.type === "cover") {
    return (
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          minHeight: 380,
          textAlign: "center",
          padding: "40px 20px",
        }}
      >
        <div style={{ fontSize: 56, marginBottom: 16 }}>🧠</div>
        <h1
          style={{
            fontFamily: "Georgia, 'Times New Roman', serif",
            fontSize: "clamp(28px, 5vw, 44px)",
            color: C.text,
            fontWeight: 700,
            margin: "0 0 12px",
            letterSpacing: -1,
          }}
        >
          {slide.title}
        </h1>
        <p style={{ color: C.muted, fontSize: 18, marginBottom: 28 }}>{slide.subtitle}</p>
        <div style={{ display: "flex", gap: 12, flexWrap: "wrap", justifyContent: "center" }}>
          {["Interactive Slides", "Code Labs", "AI Feedback", "SDG Projects", "Employer Review"].map(
            (tag) => (
              <Badge key={tag} color={C.accent}>
                {tag}
              </Badge>
            )
          )}
        </div>
        <p style={{ color: C.muted, fontSize: 13, marginTop: 28, maxWidth: 480 }}>
          Use the arrows to navigate. Every coding slide has a test zone — write the code yourself before looking at the answer.
        </p>
      </div>
    );
  }

  if (slide.type === "concept") {
    return (
      <div>
        {slide.content.explanation.split("\n\n").map((para, i) => (
          <p
            key={i}
            style={{
              color: para.startsWith("**") ? C.text : "#94a3b8",
              lineHeight: 1.8,
              fontSize: 15,
              marginBottom: 14,
            }}
            dangerouslySetInnerHTML={{
              __html: para
                .replace(/\*\*(.*?)\*\*/g, `<strong style="color:${C.text}">$1</strong>`)
                .replace(/^- /gm, "• ")
                .replace(/\n/g, "<br/>"),
            }}
          />
        ))}
        <div
          style={{
            background: "#1a1a2e",
            border: `1px solid ${C.gold}33`,
            borderRadius: 8,
            padding: "14px 18px",
            marginTop: 16,
            color: "#fde68a",
            fontSize: 14,
            fontStyle: "italic",
          }}
        >
          💡 {slide.content.analogy}
        </div>
        <EmployerNote note={slide.employerNote} />
      </div>
    );
  }

  if (slide.type === "setup") {
    return (
      <div>
        {slide.content.steps.map((s, i) => (
          <div
            key={i}
            style={{
              background: "#111827",
              border: `1px solid ${C.border}`,
              borderRadius: 8,
              padding: "14px 18px",
              marginBottom: 12,
              display: "flex",
              gap: 14,
            }}
          >
            <div
              style={{
                background: C.accent,
                color: "#000",
                fontWeight: 800,
                width: 28,
                height: 28,
                borderRadius: "50%",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                flexShrink: 0,
                fontSize: 13,
              }}
            >
              {i + 1}
            </div>
            <div>
              <div style={{ color: C.text, fontWeight: 700, marginBottom: 4 }}>{s.label}</div>
              <div style={{ color: "#94a3b8", fontSize: 13, lineHeight: 1.6 }}>{s.detail}</div>
            </div>
          </div>
        ))}
        <EmployerNote note={slide.employerNote} />
      </div>
    );
  }

  if (slide.type === "code") {
    return (
      <div>
        <EmployerNote note={slide.employerNote} />
        <div style={{ marginTop: 20 }}>
          <div style={{ color: C.muted, fontSize: 12, letterSpacing: 1, marginBottom: 6 }}>
            REFERENCE CODE
          </div>
          <CodeBlock code={slide.code} />
        </div>
        <CodeTestPanel slide={slide} />
      </div>
    );
  }

  if (slide.type === "metrics") {
    const { technical, soft, interview } = slide.content;
    return (
      <div>
        <div style={{ color: C.text, fontWeight: 700, marginBottom: 12 }}>🛠 Technical Skills</div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 24 }}>
          {technical.map((m) => (
            <div
              key={m.skill}
              style={{
                background: "#111827",
                border: `1px solid ${C.border}`,
                borderRadius: 8,
                padding: "12px 14px",
              }}
            >
              <div style={{ color: C.accent, fontWeight: 700, fontSize: 13, marginBottom: 4 }}>
                {m.skill}
              </div>
              <div style={{ color: "#64748b", fontSize: 12, lineHeight: 1.5 }}>{m.detail}</div>
            </div>
          ))}
        </div>

        <div style={{ color: C.text, fontWeight: 700, marginBottom: 12 }}>🤝 Soft Skills</div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 24 }}>
          {soft.map((m) => (
            <div
              key={m.skill}
              style={{
                background: "#111827",
                border: `1px solid ${C.border}`,
                borderRadius: 8,
                padding: "12px 14px",
              }}
            >
              <div style={{ color: C.gold, fontWeight: 700, fontSize: 13, marginBottom: 4 }}>
                {m.skill}
              </div>
              <div style={{ color: "#64748b", fontSize: 12, lineHeight: 1.5 }}>{m.detail}</div>
            </div>
          ))}
        </div>

        <div style={{ color: C.text, fontWeight: 700, marginBottom: 12 }}>
          🎤 Common Interview Questions
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {interview.map((q, i) => (
            <div
              key={i}
              style={{
                background: "#0f1a2e",
                border: `1px solid ${C.border}`,
                borderLeft: `3px solid ${C.rose}`,
                borderRadius: 6,
                padding: "10px 14px",
                color: "#94a3b8",
                fontSize: 13,
              }}
            >
              Q{i + 1}: {q}
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (slide.type === "projects") {
    return (
      <div>
        <p style={{ color: C.muted, fontSize: 14, marginBottom: 20, lineHeight: 1.7 }}>
          These projects are designed to impress employers AND make a real-world impact aligned with UN Sustainable Development Goals. Click any project to get a step-by-step walkthrough.
        </p>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
          {SDG_PROJECTS.map((p) => (
            <div
              key={p.id}
              onClick={() => setActiveProject(p)}
              style={{
                background: "#111827",
                border: `1px solid ${C.border}`,
                borderRadius: 12,
                padding: 20,
                cursor: "pointer",
                transition: "border-color 0.2s",
              }}
              onMouseEnter={(e) => (e.currentTarget.style.borderColor = C.accent)}
              onMouseLeave={(e) => (e.currentTarget.style.borderColor = C.border)}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
                <span style={{ fontSize: 28 }}>{p.icon}</span>
                <Badge color={C.accent} style={{ fontSize: 10 }}>
                  {p.sdg}
                </Badge>
              </div>
              <div style={{ color: C.text, fontWeight: 700, fontSize: 15, marginBottom: 6 }}>
                {p.title}
              </div>
              <div style={{ color: C.muted, fontSize: 12, marginBottom: 10 }}>{p.tagline}</div>
              <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                {p.skills.slice(0, 3).map((s) => (
                  <Badge key={s} color={C.gold}>
                    {s}
                  </Badge>
                ))}
              </div>
              <div
                style={{
                  marginTop: 14,
                  color: C.accent,
                  fontSize: 12,
                  fontWeight: 700,
                  letterSpacing: 0.5,
                }}
              >
                View Walkthrough →
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (slide.type === "review") {
    return <NotebookReview />;
  }

  return null;
}

// ── MAIN APP ───────────────────────────────────────────────────
export default function App() {
  const [current, setCurrent] = useState(0);
  const [activeProject, setActiveProject] = useState(null);
  const total = SLIDES.length;

  const slide = SLIDES[current];
  const progress = ((current + 1) / total) * 100;

  const goPrev = () => {
    setActiveProject(null);
    setCurrent((c) => Math.max(0, c - 1));
  };
  const goNext = () => {
    setActiveProject(null);
    setCurrent((c) => Math.min(total - 1, c + 1));
  };

  return (
    <div
      style={{
        background: C.bg,
        minHeight: "100vh",
        fontFamily: "'IBM Plex Sans', 'Segoe UI', sans-serif",
        color: C.text,
        display: "flex",
        flexDirection: "column",
      }}
    >
      {/* Top bar */}
      <div
        style={{
          background: C.panel,
          borderBottom: `1px solid ${C.border}`,
          padding: "12px 24px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 16,
          flexWrap: "wrap",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ fontSize: 22 }}>🧠</span>
          <span
            style={{
              fontFamily: "Georgia, serif",
              fontWeight: 700,
              fontSize: 16,
              color: C.accent,
              letterSpacing: -0.5,
            }}
          >
            ML Mastery
          </span>
        </div>

        {/* Nav pills */}
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {SLIDES.map((s, i) => (
            <button
              key={s.id}
              onClick={() => {
                setActiveProject(null);
                setCurrent(i);
              }}
              title={s.title}
              style={{
                background: i === current ? C.accent : "#1e2d45",
                color: i === current ? "#000" : C.muted,
                border: "none",
                borderRadius: 4,
                width: 28,
                height: 28,
                fontWeight: 700,
                fontSize: 11,
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              {i + 1}
            </button>
          ))}
        </div>

        <div style={{ color: C.muted, fontSize: 12 }}>
          {current + 1} / {total}
        </div>
      </div>

      {/* Progress bar */}
      <div style={{ height: 3, background: C.border }}>
        <div
          style={{
            height: "100%",
            width: `${progress}%`,
            background: `linear-gradient(90deg, ${C.accent}, ${C.gold})`,
            transition: "width 0.4s ease",
          }}
        />
      </div>

      {/* Main */}
      <div style={{ flex: 1, maxWidth: 860, margin: "0 auto", width: "100%", padding: "30px 24px 100px" }}>
        {/* Slide header */}
        {slide.type !== "cover" && (
          <div style={{ marginBottom: 20 }}>
            <div
              style={{
                color: C.muted,
                fontSize: 11,
                letterSpacing: 2,
                textTransform: "uppercase",
                marginBottom: 6,
              }}
            >
              Slide {current + 1} — {slide.type}
            </div>
            <h2
              style={{
                fontFamily: "Georgia, serif",
                fontSize: "clamp(20px, 4vw, 30px)",
                fontWeight: 700,
                margin: 0,
                color: C.text,
                letterSpacing: -0.5,
              }}
            >
              {slide.title}
            </h2>
          </div>
        )}

        {/* Slide content or project walkthrough */}
        {activeProject && slide.type === "projects" ? (
          <ProjectWalkthrough project={activeProject} onBack={() => setActiveProject(null)} />
        ) : (
          <SlideContent slide={slide} setActiveProject={setActiveProject} />
        )}
      </div>

      {/* Bottom nav */}
      <div
        style={{
          position: "fixed",
          bottom: 0,
          left: 0,
          right: 0,
          background: C.panel,
          borderTop: `1px solid ${C.border}`,
          padding: "14px 24px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          zIndex: 100,
        }}
      >
        <button
          onClick={goPrev}
          disabled={current === 0}
          style={{
            background: current === 0 ? "#1e2d45" : "#1e2d45",
            color: current === 0 ? C.border : C.text,
            border: `1px solid ${C.border}`,
            borderRadius: 8,
            padding: "9px 22px",
            fontWeight: 700,
            fontSize: 14,
            cursor: current === 0 ? "not-allowed" : "pointer",
          }}
        >
          ← Previous
        </button>

        <div style={{ color: C.muted, fontSize: 12, textAlign: "center" }}>
          {slide.title}
        </div>

        <button
          onClick={goNext}
          disabled={current === total - 1}
          style={{
            background: current === total - 1 ? "#1e2d45" : C.accent,
            color: current === total - 1 ? C.border : "#000",
            border: "none",
            borderRadius: 8,
            padding: "9px 22px",
            fontWeight: 700,
            fontSize: 14,
            cursor: current === total - 1 ? "not-allowed" : "pointer",
          }}
        >
          Next →
        </button>
      </div>
    </div>
  );
}
