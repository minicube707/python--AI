import os
import shutil

module_dir = os.path.dirname(__file__)
os.chdir(module_dir)

def remove_pycache(start_dir="."):
    """
    Supprime récursivement tous les dossiers __pycache__ à partir de start_dir.
    """
    removed_count = 0

    for root, dirs, files in os.walk(start_dir):
        for dir_name in dirs:
            if dir_name == "__pycache__":
                full_path = os.path.join(root, dir_name)
                try:
                    shutil.rmtree(full_path)
                    removed_count += 1
                    print(f"✅ Supprimé : {full_path}")
                except Exception as e:
                    print(f"❌ Erreur lors de la suppression de {full_path} : {e}")

    if removed_count == 0:
        print("Aucun dossier __pycache__ trouvé.")
    else:
        print(f"\n🧹 {removed_count} dossier(s) __pycache__ supprimé(s).")

if __name__ == "__main__":
    # Tu peux remplacer "." par un autre chemin si tu veux cibler un dossier précis.
    remove_pycache(module_dir)
