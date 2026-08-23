"""
Results & Comparison Script — Triphala Component Classification
================================================================
This script loads all trained models, evaluates them on the dataset,
and generates:
  1. Confusion Matrix for each model
  2. Classification Report (Precision, Recall, F1-Score)
  3. Accuracy & Loss Comparison Bar Chart
  4. Summary Table saved as an image

Usage:
  python results_comparison.py

Note: Train the models first using Model1.py, Resnet.py, VGG16.py, VGG19 (1).py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# ========================== Configuration ==========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "Resized")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
CLASS_NAMES = ['amla', 'bherda', 'herda']
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Model paths and their preprocessing functions
MODELS_CONFIG = {
    'Custom CNN': {
        'path': os.path.join(SCRIPT_DIR, 'final_custom_model.keras'),
        'preprocess': None,  # Uses rescale=1/255
    },
    'ResNet50': {
        'path': os.path.join(SCRIPT_DIR, 'resnet50_fine_tuned_model.keras'),
        'preprocess': resnet_preprocess,
    },
    'VGG16': {
        'path': os.path.join(SCRIPT_DIR, 'vgg16_fine_tuned_model.keras'),
        'preprocess': vgg16_preprocess,
    },
    'VGG19': {
        'path': os.path.join(SCRIPT_DIR, 'vgg19_fine_tuned_model.keras'),
        'preprocess': vgg19_preprocess,
    },
}

# ========================== Helper Functions ==========================

def create_data_generator(preprocess_fn):
    """Create a data generator with the appropriate preprocessing."""
    if preprocess_fn is None:
        datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.15)
    else:
        datagen = ImageDataGenerator(preprocessing_function=preprocess_fn, validation_split=0.15)

    generator = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        classes=CLASS_NAMES,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    return generator


def evaluate_model(model, generator):
    """Evaluate a model and return predictions, true labels, loss, and accuracy."""
    generator.reset()
    loss, accuracy = model.evaluate(generator, verbose=0)

    generator.reset()
    predictions = model.predict(generator, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = generator.classes

    return {
        'loss': loss,
        'accuracy': accuracy,
        'predicted': predicted_classes,
        'true': true_classes,
        'probabilities': predictions,
    }


def plot_confusion_matrix(true_labels, pred_labels, class_names, model_name, save_path):
    """Plot and save a confusion matrix."""
    cm = confusion_matrix(true_labels, pred_labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(f'Confusion Matrix — {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_accuracy_comparison(results, save_path):
    """Plot accuracy and loss comparison bar chart."""
    model_names = list(results.keys())
    accuracies = [results[m]['accuracy'] * 100 for m in model_names]
    losses = [results[m]['loss'] for m in model_names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Accuracy comparison
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
    bars1 = ax1.bar(model_names, accuracies, color=colors[:len(model_names)], edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 105)
    ax1.grid(axis='y', alpha=0.3)
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1,
                 f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Loss comparison
    bars2 = ax2.bar(model_names, losses, color=colors[:len(model_names)], edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Model Loss Comparison', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    for bar, loss in zip(bars2, losses):
        ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                 f'{loss:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_summary_table(results, save_path):
    """Plot a summary table as an image."""
    model_names = list(results.keys())

    # Table data
    headers = ['Model', 'Accuracy (%)', 'Loss', 'Precision', 'Recall', 'F1-Score']
    table_data = []
    for name in model_names:
        r = results[name]
        report = classification_report(r['true'], r['predicted'],
                                       target_names=CLASS_NAMES, output_dict=True)
        table_data.append([
            name,
            f"{r['accuracy'] * 100:.2f}",
            f"{r['loss']:.4f}",
            f"{report['weighted avg']['precision']:.4f}",
            f"{report['weighted avg']['recall']:.4f}",
            f"{report['weighted avg']['f1-score']:.4f}",
        ])

    # Find best model
    best_idx = np.argmax([results[m]['accuracy'] for m in model_names])

    fig, ax = plt.subplots(figsize=(12, 2 + len(model_names) * 0.6))
    ax.axis('off')
    ax.set_title('Model Comparison Summary', fontsize=16, fontweight='bold', pad=20)

    table = ax.table(cellText=table_data, colLabels=headers,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)

    # Style header
    for j in range(len(headers)):
        table[0, j].set_facecolor('#2196F3')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Highlight best model row
    for j in range(len(headers)):
        table[best_idx + 1, j].set_facecolor('#E8F5E9')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ========================== Main ==========================

def main():
    print("=" * 60)
    print("  Triphala Component Classification — Results Comparison")
    print("=" * 60)

    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Check which models are available
    available_models = {}
    for name, config in MODELS_CONFIG.items():
        if os.path.exists(config['path']):
            available_models[name] = config
            print(f"  ✅ Found: {name} ({os.path.basename(config['path'])})")
        else:
            print(f"  ❌ Not found: {name} — train it first")

    if not available_models:
        print("\n⚠️  No trained models found! Run the training scripts first:")
        print("    python Model1.py")
        print("    python Resnet.py")
        print("    python VGG16.py")
        print('    python "VGG19 (1).py"')
        return

    print(f"\nEvaluating {len(available_models)} model(s)...\n")

    # Evaluate each model
    results = {}
    for name, config in available_models.items():
        print(f"📊 Evaluating {name}...")
        model = tf.keras.models.load_model(config['path'])
        generator = create_data_generator(config['preprocess'])
        results[name] = evaluate_model(model, generator)
        print(f"   Accuracy: {results[name]['accuracy'] * 100:.2f}%  |  Loss: {results[name]['loss']:.4f}")

        # Confusion Matrix
        cm_path = os.path.join(RESULTS_DIR, f"confusion_matrix_{name.lower().replace(' ', '_')}.png")
        plot_confusion_matrix(results[name]['true'], results[name]['predicted'],
                              CLASS_NAMES, name, cm_path)

        # Classification Report
        report = classification_report(results[name]['true'], results[name]['predicted'],
                                       target_names=CLASS_NAMES)
        report_path = os.path.join(RESULTS_DIR, f"classification_report_{name.lower().replace(' ', '_')}.txt")
        with open(report_path, 'w') as f:
            f.write(f"Classification Report — {name}\n")
            f.write("=" * 50 + "\n\n")
            f.write(report)
        print(f"  Saved: {report_path}")
        print()

    # Comparison charts (only if more than 1 model)
    if len(results) >= 1:
        print("📈 Generating comparison charts...")
        comparison_path = os.path.join(RESULTS_DIR, "accuracy_loss_comparison.png")
        plot_accuracy_comparison(results, comparison_path)

        summary_path = os.path.join(RESULTS_DIR, "summary_table.png")
        plot_summary_table(results, summary_path)

    # Print final summary
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<15} {'Accuracy':>10} {'Loss':>10}")
    print("  " + "-" * 37)
    best_model = None
    best_acc = 0
    for name, r in results.items():
        acc = r['accuracy'] * 100
        marker = ""
        if acc > best_acc:
            best_acc = acc
            best_model = name
        print(f"  {name:<15} {acc:>9.2f}% {r['loss']:>10.4f}")

    print("  " + "-" * 37)
    print(f"  🏆 Best Model: {best_model} ({best_acc:.2f}%)")
    print(f"\n  All results saved in: {RESULTS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
