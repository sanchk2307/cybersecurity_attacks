import subprocess
import zipfile
from pathlib import Path


class GetFiles():
    """
    A utility class for downloading and extracting files from remote URLs.

    This class handles downloading files using curl, automatically extracting
    ZIP archives, and managing the data directory structure. It is designed
    to download multiple files from a dictionary of filename-URL pairs.

    Attributes:
        data_dir (Path): The directory where files will be downloaded and extracted.
        files (dict): Dictionary mapping filenames to their download URLs.
    """

    def __init__(self, files_to_download: dict, data_directory: Path) -> None:
        """
        Initialize the GetFiles class and begin the download process.

        Creates the data directory if it doesn't exist and automatically starts
        downloading all files specified in the files_to_download dictionary.

        Args:
            files_to_download (dict): Dictionary where keys are filenames and values
                are URLs to download from.
            data_directory (Path): Path object representing the directory where files
                should be saved.

        Returns:
            None

        Example:
            files = {
                'data.csv': 'https://example.com/data.csv',
                'archive.zip': 'https://example.com/archive.zip'
            }
            downloader = GetFiles(files, Path('data'))
        """
        # Create data directory
        self.data_dir = data_directory
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.files = files_to_download
        
        self._get_files()

    def _get_files(self) -> None:
        """
        Download files from URLs and extract ZIP archives.

        This private method iterates through all files in the files dictionary,
        downloads each file using curl, and automatically extracts any ZIP files.
        After extraction, the original ZIP file is removed to save space.
        Displays a summary of all downloaded files with their sizes.

        Args:
            None

        Returns:
            None

        Raises:
            subprocess.CalledProcessError: If curl download fails, the error is caught
                and logged, but execution continues for remaining files.

        Side Effects:
            - Downloads files to self.data_dir
            - Extracts ZIP archives
            - Removes ZIP files after extraction
            - Prints download progress and file listing
        """
        for filename, url in self.files.items():
            filepath = self.data_dir / filename
            
            # Download with curl
            print(f"Downloading {filename}...")
            try:
                subprocess.run(['curl', '-L', '-o', str(filepath), url], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Failed to download {filename} from {url}. Error: {e}")
                continue
            
            # Unzip if needed
            if filename.endswith('.zip'):
                print(f"Extracting {filename}...")
                with zipfile.ZipFile(filepath, 'r') as z:
                    z.extractall(self.data_dir)
                filepath.unlink()  # remove zip after extraction

        print(f"\nDone! Files in {self.data_dir}:")
        for f in self.data_dir.iterdir():
            print(f"  {f.name} ({f.stat().st_size / 1e6:.1f} MB)")