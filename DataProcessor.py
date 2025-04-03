import pandas as pd

class DataProcessor:
    def __init__(self, df: pd.DataFrame):
        """
        Initializes the DataProcessor with a Pandas DataFrame.

        :param df: Pandas DataFrame to process
        """
        self.df = df

    def display_info(self):
        """Displays the number of features and instances in the dataset."""
        print('Number of Features In Dataset:', self.df.shape[1])
        print('Number of Instances In Dataset:', self.df.shape[0])

    def drop_name_column(self):
        """Drops the 'name' column if it exists."""
        if 'name' in self.df.columns:
            self.df.drop(columns=['name'], inplace=True)
            print("'name' column dropped.")
        else:
            print("No 'name' column found.")

    def convert_target(self):
        """Converts the 'target' column to uint8 if it exists."""
        if 'target' in self.df.columns:
            self.df['target'] = self.df['target'].astype('uint8')
            print("'target' column converted to uint8.")
        else:
            print("No 'target' column found.")

    def check_duplicates(self):
        """Checks for duplicate rows in the dataset and prints the count."""
        duplicate_count = self.df.duplicated().sum()
        print('Number of Duplicated Rows:', duplicate_count)

    def process(self):
        """Runs all processing steps on the dataset."""
        self.display_info()
        # self.drop_name_column()
        # self.display_info()
        self.convert_target()
        self.check_duplicates()

        return self.df