def get_user_input(prompt_message):
    return input(prompt_message)

def prompt_add_documents():
    while True:
        choice = input("Do you want to add documents to the vector database? (yes/no): ").strip().lower()
        if choice in ['yes', 'y']:
            return True
        elif choice in ['no', 'n']:
            return False
        else:
            print("Please enter 'yes' or 'no'.")

def prompt_continue_conversation():
    while True:
        choice = input("Do you want to continue the conversation? (yes/no): ").strip().lower()
        if choice in ['yes', 'y']:
            return True
        elif choice in ['no', 'n']:
            return False
        else:
            print("Please enter 'yes' or 'no'.")