import os


def create_windows_structure():
    print("Creating project structure for Windows...")

    # Create directories
    directories = [
        'static\\css',
        'static\\js',
        'static\\maps',
        'templates',
        'uploads',
        'reports',
        'training_scripts'
    ]

    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✓ Created directory: {directory}")
        except Exception as e:
            print(f"✗ Error creating {directory}: {e}")

    # Create main files
    main_files = [
        'web_dashboard.py',
        'config.py',
        'camera_manager.py',
        'detection_processor.py',
        'report_generator.py',
        'data_manager.py',
        'requirements.txt',
        'road_defect_final_model.pt'
    ]

    for file in main_files:
        try:
            with open(file, 'w') as f:
                f.write('')
            print(f"✓ Created file: {file}")
        except Exception as e:
            print(f"✗ Error creating {file}: {e}")

    # Create template files
    template_files = [
        'templates\\index.html',
        'templates\\dashboard.html'
    ]

    for file in template_files:
        try:
            with open(file, 'w') as f:
                f.write('')
            print(f"✓ Created file: {file}")
        except Exception as e:
            print(f"✗ Error creating {file}: {e}")

    # Create static files
    static_files = [
        'static\\css\\styles.css',
        'static\\js\\dashboard.js'
    ]

    for file in static_files:
        try:
            with open(file, 'w') as f:
                f.write('')
            print(f"✓ Created file: {file}")
        except Exception as e:
            print(f"✗ Error creating {file}: {e}")

    # Create training script files
    training_files = [
        'training_scripts\\boxchecktest.py',
        'training_scripts\\check_dataset.py',
        'training_scripts\\CheckYamlDataFile.py',
        'training_scripts\\Convert_polygon_to_Bounding_Box.py',
        'training_scripts\\create_html.py',
        'training_scripts\\CreateDataYaml.py',
        'training_scripts\\data.yaml',
        'training_scripts\\DEletingOldData.py',
        'training_scripts\\detection_result.jpg'
    ]

    for file in training_files:
        try:
            with open(file, 'w') as f:
                f.write('')
            print(f"✓ Created file: {file}")
        except Exception as e:
            print(f"✗ Error creating {file}: {e}")

    print("\nProject structure created successfully!")
    print("Now add content to the key files.")


if __name__ == '__main__':
    create_windows_structure()

