# 🛠️ Adding a New Feature Transformer

Welcome to the feature engineering guide! Our pipeline is built on a custom, scikit-learn-inspired DataFrame-oriented architecture. This means you get **automatic schema validation**, **fitted-state checking**, and **automatic feature name inference** for free.

This guide will walk you through adding a new transformer in 3 easy steps.

---

## 📋 The 3-Step Process

1. **Create** your transformer class (Stateless or Stateful).
2. **Register** it in the pipeline builder.
3. **Run** the pipeline!

---

## Step 1: Create Your Transformer

First, decide if your transformer needs to "learn" something from the training data.
* **Stateless**: Just applies a math/string operation (e.g., text length, multiplying columns).
* **Stateful**: Needs to calculate statistics from the training data (e.g., mean imputation, scaling, encoding).

Choose the template below that fits your needs, copy it into a new file (e.g., `my_new_transformer.py`), and fill in your logic!

### 🟢 Option A: Stateless Transformer Template
*Use this when your transformation doesn't depend on the training data.*

```python
import pandas as pd
from .base import BaseFeatureTransformer

class TextLengthTransformer(BaseFeatureTransformer):
    """Calculates the character length of a text column."""
    
    def __init__(self, text_column: str, output_column: str = "text_length"):
        # 1. ALWAYS call super().__init__() to set up base tracking
        super().__init__()
        
        # 2. Save your configuration
        self.text_column = text_column
        self.output_column = output_column

    def _fit_transformer(self, X: pd.DataFrame, y: pd.Series | None) -> None:
        """
        Learn state from training data. 
        Leave this empty for stateless transformers!
        """
        pass

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the transformation. 
        MUST return a pandas DataFrame.
        """
        # Pro-tip: Use .copy() to avoid mutating the original DataFrame!
        X_out = X.copy()
        
        # --- YOUR LOGIC HERE ---
        X_out[self.output_column] = X_out[self.text_column].apply(len)
        
        return X_out
```

### 🔵 Option B: Stateful Transformer Template
*Use this when you need to calculate statistics (like mean, std, or vocabularies) during `fit()` and use them during `transform()`.*

```python
import pandas as pd
from .base import BaseFeatureTransformer

class MeanCenterTransformer(BaseFeatureTransformer):
    """Subtracts the training mean from a numeric column."""
    
    def __init__(self, column: str):
        # 1. ALWAYS call super().__init__()
        super().__init__()
        self.column = column
        
        # 2. Initialize your state variables (conventionally ending with an underscore)
        self.mean_ = None 

    def _fit_transformer(self, X: pd.DataFrame, y: pd.Series | None) -> None:
        """
        Learn state from the training data.
        This runs ONLY during .fit() or .fit_transform().
        """
        # --- YOUR FITTING LOGIC HERE ---
        self.mean_ = X[self.column].mean()

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the transformation using the learned state.
        MUST return a pandas DataFrame.
        """
        X_out = X.copy()
        
        # --- YOUR TRANSFORMATION LOGIC HERE ---
        # Use the state calculated during fit!
        X_out[f"{self.column}_centered"] = X_out[self.column] - self.mean_
        
        return X_out
```

---

## Step 2: Register the Transformer

Now that your transformer is built, you need to plug it into the pipeline. Open `builder.py` and add your new class to the `transformers` list.

```python
# builder.py

from .base import FeaturePipeline
# 👇 1. Import your new transformer(s) here
from .my_new_transformer import TextLengthTransformer, MeanCenterTransformer 

def get_feature_pipeline() -> FeaturePipeline:
    """Build and return the project's feature-engineering pipeline."""
    
    transformers = [
        # 👇 2. Instantiate and add them to this list in execution order!
        TextLengthTransformer(text_column="description"),
        MeanCenterTransformer(column="calories"),
    ]
    
    if not transformers:
        raise RuntimeError(
            "The feature pipeline has no configured transformers. "
            "Add feature transformers before calling get_feature_pipeline()."
        )
        
    return FeaturePipeline(transformers)
```

---

## Step 3: Verify It Works!

You can quickly test your new pipeline in a Python script or Jupyter Notebook to ensure everything connects properly:

```python
import pandas as pd
from builder import get_feature_pipeline

# 1. Create some dummy data
X_train = pd.DataFrame({
    "description": ["apple", "banana bread", "cherry"],
    "calories": [50, 150, 60]
})

# 2. Build and fit the pipeline
pipeline = get_feature_pipeline()
X_engineered = pipeline.fit_transform(X_train)

# 3. Check the results
print(X_engineered)
print("\nOutput Features:", pipeline.get_feature_names_out())
```

---

## 💡 Golden Rules & Pro-Tips

1. **Always return a DataFrame**: Your `_transform()` method **must** return a `pandas.DataFrame`. If it returns a Series, numpy array, or list, the base class will throw a `TypeError`.
2. **Don't mutate the original data**: Always use `X.copy()` inside `_transform()` before modifying it. This prevents weird bugs where training data gets accidentally overwritten.
3. **Feature names are free!**: You **do not** need to write `get_feature_names_out()`. The `BaseFeatureTransformer` automatically looks at the columns of the DataFrame returned by your `_transform()` method and records them.
4. **Missing columns are caught automatically**: If your transformer requires a column named `"calories"`, and the user passes data without it, the base class will automatically raise a clear `ValueError` during `transform()`.
5. **Naming Convention**: For stateful variables learned during `fit()`, follow the scikit-learn convention of ending the attribute name with an underscore (e.g., `self.mean_`, `self.vocabulary_`).
