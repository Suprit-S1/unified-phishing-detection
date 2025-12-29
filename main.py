import sys
from src.train_model import train
from src.predict import interactive_mode, predict_single


def main():

    while True:
        print("\n===============================================")
        print("   Unified Multi-Channel Phishing Detection    ")
        print("===============================================")
        print("1. Train Model (Process Data -> Train -> Save)")
        print("2. Test Model (Detailed Interactive Mode)")
        print("3. Quick Prediction (Direct Input)")
        print("4. Exit")

        choice = input("\nSelect an option (1-4): ").strip()

        if choice == '1':
            # Run the training pipeline
            train()

        elif choice == '2':
            # Launch the detailed sub-menu (Subject/Body separation)
            interactive_mode()

        elif choice == '3':
            # Quick Prediction: Just take input and show result
            text = input("\nEnter text to classify: ")
            if text:
                label, conf = predict_single(text)
                print(f"Analysis: {label} ({conf:.1%})")
                input("\nPress Enter to continue...")

        elif choice == '4':
            print("Exiting...")
            sys.exit()

        else:
            print("Invalid choice. Please select 1-4.")


if __name__ == "__main__":
    # Support for command line arguments (e.g., python main.py "Suspicious link")
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        label, conf = predict_single(text)
        print(f"Input: {text}")
        print(f"Result: {label} ({conf:.1%})")
    else:
        main()