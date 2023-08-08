import os

def get_file_emoji(file_extension):
    emoji_dict = {
        '.py': '🐍',
        '.blend': '💻',
        '.md': '📝',
        '.txt': '📄',
        '.sh': '🔧'
        # Add more extensions and emojis as needed
    }
    return emoji_dict.get(file_extension, '📁')  # Default to folder emoji

def is_relevant_item(item_name):
    included_extensions = ['.py', '.blend', '.md', '.txt', '.sh']
    return any(item_name.endswith(ext) for ext in included_extensions)

def is_relevant_folder(folder_name):
    excluded_folders = ['smoke_generator_env', '.git', '__pycache__', 'harmonization_scripts']
    return folder_name not in excluded_folders

def print_project_architecture(base_folder, indent=0):
    if not os.path.exists(base_folder):
        print(f"Folder '{base_folder}' not found.")
        return

    for item in os.listdir(base_folder):
        item_path = os.path.join(base_folder, item)
        if os.path.isdir(item_path):
            if is_relevant_folder(item):
                print(" " * indent + f"{get_file_emoji('📁')} {item}")
                # Recursive call to print only the relevant items
                print_project_architecture(item_path, indent + 2)
        elif is_relevant_item(item):
            file_extension = os.path.splitext(item)[1]
            emoji = get_file_emoji(file_extension)
            print(" " * indent + f"{emoji} {item}")

# Replace 'your_project_folder_path' with the path of your project's base folder
project_base_folder = r'D:\mploi\Documents\Albatros\albatros'
print_project_architecture(project_base_folder)
