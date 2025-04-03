from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class TrainTestSplit:
    def __init__(self, df, target_column='price', test_size=0.2, random_state=42, smote_random_state=300):
        """
        Initialize the TrainTestSplit with a dataframe and parameters.

        :param df: Pandas DataFrame containing features and target.
        :param target_column: The name of the target column.
        :param test_size: Proportion of the dataset to be used for testing.
        :param random_state: Random state for reproducibility in train-test split.
        :param smote_random_state: Random state for SMOTE balancing.
        """
        self.df = df
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.smote_random_state = smote_random_state

    def balance_and_split(self):
        """
        Balances the dataset using SMOTE, normalizes features, and splits into training and testing sets.

        :return: X_train, X_test, y_train, y_test
        """
        # Separate features and target
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]

        print('Feature (X) Shape Before Balancing:', X.shape)
        print('Target (y) Shape Before Balancing:', y.shape)

        # Count samples for each class
        class_counts = y.value_counts()
        print("Class distribution before balancing:", class_counts.to_dict())

        # Check min samples and apply appropriate resampling
        min_samples = class_counts.min()

        if min_samples >= 6:
            # Use regular SMOTE if enough samples
            print("Using regular SMOTE with k_neighbors=5")
            sm = SMOTE(random_state=self.smote_random_state)
        elif min_samples > 1:
            # Use SMOTE with reduced k_neighbors
            k = min(min_samples - 1, 5)  # k must be at least 1 and at most 5
            print(f"Using SMOTE with k_neighbors={k} due to small class size")
            sm = SMOTE(k_neighbors=k, random_state=self.smote_random_state)
        else:
            # Fallback to RandomOverSampler if classes have only 1 sample
            print("Using RandomOverSampler instead of SMOTE due to classes with single samples")
            sm = RandomOverSampler(random_state=self.smote_random_state)

        X, y = sm.fit_resample(X, y)

        print('Feature (X) Shape After Balancing:', X.shape)
        print('Target (y) Shape After Balancing:', y.shape)

        # Scale features between -1 and 1
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_scaled = scaler.fit_transform(X)

        # Split dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=self.test_size, random_state=self.random_state
        )

        return X_train, X_test, y_train, y_test