import os
import csv
import pandas as pd
from sklearn.linear_model import LinearRegression

DATA_FILE = "usage_data.csv"
COLUMNS = ["id", "household_size", "previous_month_units", "month", "current_month_units"]


def initialize_file():
    if not os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(COLUMNS)
        except IOError as e:
            print(f"Error creating data file: {e}")


def load_data():
    try:
        df = pd.read_csv(DATA_FILE)
        for col in COLUMNS:
            if col not in df.columns:
                df[col] = pd.Series(dtype="float")
        return df
    except FileNotFoundError:
        print("Data file not found. Creating a new one...")
        initialize_file()
        return pd.DataFrame(columns=COLUMNS)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=COLUMNS)
    except Exception as e:
        print(f"Unexpected error while loading data: {e}")
        return pd.DataFrame(columns=COLUMNS)


def save_data(df):
    try:
        df.to_csv(DATA_FILE, index=False)
        print("Data saved successfully.\n")
    except IOError as e:
        print(f"Error saving data: {e}")


def get_next_id(df):
    if df.empty:
        return 1
    try:
        return int(df["id"].max()) + 1
    except (ValueError, TypeError):
        return len(df) + 1


def get_float_input(prompt):
    while True:
        try:
            value = float(input(prompt))
            if value < 0:
                print("Value cannot be negative. Try again.")
                continue
            return value
        except ValueError:
            print("Invalid input. Please enter a numeric value.")


def get_nonempty_string(prompt):
    while True:
        value = input(prompt).strip()
        if value:
            return value
        print("Input cannot be empty. Try again.")


def add_record(df):
    print("\n--- Add New Usage Record ---")
    try:
        household_size = get_float_input("Household size (number of members): ")
        previous_units = get_float_input("Previous month's units consumed: ")
        month = get_nonempty_string("Month (e.g., January): ")
        current_units = get_float_input("Current month's units consumed: ")

        new_row = {
            "id": get_next_id(df),
            "household_size": household_size,
            "previous_month_units": previous_units,
            "month": month,
            "current_month_units": current_units,
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        save_data(df)
    except Exception as e:
        print(f"Error adding record: {e}")
    return df


def view_records(df):
    print("\n--- All Usage Records ---")
    if df.empty:
        print("No records found.\n")
    else:
        print(df.to_string(index=False))
        print()


def update_record(df):
    print("\n--- Update Usage Record ---")
    if df.empty:
        print("No records available to update.\n")
        return df
    try:
        view_records(df)
        record_id = int(input("Enter the ID of the record to update: "))
        if record_id not in df["id"].values:
            print("Record ID not found.\n")
            return df

        index = df.index[df["id"] == record_id][0]

        print("Leave a field blank to keep its current value.")
        household_size = input(f"Household size [{df.at[index, 'household_size']}]: ").strip()
        previous_units = input(f"Previous month units [{df.at[index, 'previous_month_units']}]: ").strip()
        month = input(f"Month [{df.at[index, 'month']}]: ").strip()
        current_units = input(f"Current month units [{df.at[index, 'current_month_units']}]: ").strip()

        if household_size:
            df.at[index, "household_size"] = float(household_size)
        if previous_units:
            df.at[index, "previous_month_units"] = float(previous_units)
        if month:
            df.at[index, "month"] = month
        if current_units:
            df.at[index,"current_month_units"] = float(current_units)

        save_data(df)
    except ValueError:
        print("Invalid input format. Update cancelled.\n")
    except Exception as e:
        print(f"Error updating record: {e}")
    return df


def predict_usage(df):
    print("\n--- Predict Next Usage (ML Model) ---")
    try:
        clean_df = df.dropna(subset=["household_size", "previous_month_units", "current_month_units"])
        if len(clean_df) < 2:
            print("Not enough data to train a model. Add at least 2 complete records first.\n")
            return

        X = clean_df[["household_size", "previous_month_units"]]
        y = clean_df["current_month_units"]

        model = LinearRegression()
        model.fit(X, y)

        household_size = get_float_input("Enter household size for prediction: ")
        previous_units = get_float_input("Enter previous month's units for prediction: ")

        input_df = pd.DataFrame(
            [[household_size, previous_units]],
            columns=["household_size", "previous_month_units"],
        )
        prediction = model.predict(input_df)[0]
        prediction = max(0, prediction)
        print(f"\nPredicted utility usage for next month: {prediction:.2f} units\n")
    except Exception as e:
        print(f"Error during prediction: {e}")


def display_menu():
    print("=" * 45)
    print(" UTILITY USAGE PREDICTION TOOL")
    print("=" * 45)
    print("1. Add usage record")
    print("2. View all records")
    print("3. Update a record")
    print("4. Predict next month's usage (ML)")
    print("5. Exit")
    print("=" * 45)


def main():
    initialize_file()
    df = load_data()

    while True:
        display_menu()
        try:
            choice = input("Enter your choice (1-5): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting program. Goodbye!")
            break

        if choice == "1":
            df = add_record(df)
        elif choice == "2":
            view_records(df)
        elif choice == "3":
            df = update_record(df)
        elif choice == "4":
            predict_usage(df)
        elif choice == "5":
            print("Exiting program. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 5.\n")


if __name__ == "__main__":
    main()