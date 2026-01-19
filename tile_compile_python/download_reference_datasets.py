from validation.reference_datasets import ReferenceDatasetManager

def download_reference_datasets():
    """
    Download comprehensive reference astronomical datasets
    """
    # Initialize dataset manager
    manager = ReferenceDatasetManager()
    
    # Reference astronomical targets with coordinates
    targets = [
        # Galaxy targets
        {'name': 'M31', 'ra': 10.6847, 'dec': 41.2692},  # Andromeda Galaxy
        {'name': 'M51', 'ra': 202.4691, 'dec': 47.1951},  # Whirlpool Galaxy
        {'name': 'M101', 'ra': 210.8028, 'dec': 54.3493},  # Pinwheel Galaxy
        
        # Nebula targets
        {'name': 'M42', 'ra': 83.8223, 'dec': -5.3911},   # Orion Nebula
        {'name': 'M57', 'ra': 283.4764, 'dec': 33.0363},  # Ring Nebula
        {'name': 'M17', 'ra': 274.7, 'dec': -16.1833},    # Omega Nebula
        
        # Star clusters
        {'name': 'M45', 'ra': 56.8725, 'dec': 24.1167},   # Pleiades
        {'name': 'M13', 'ra': 250.4233, 'dec': 36.4600}   # Hercules Globular Cluster
    ]
    
    # Download SDSS and Hubble images
    all_images = []
    
    for target in targets:
        print(f"Processing target: {target['name']}")
        
        # Download SDSS images
        sdss_images = manager.download_sdss_images(
            ra=target['ra'], 
            dec=target['dec'], 
            radius=0.5
        )
        
        # Download Hubble images
        hubble_images = manager.download_hubble_images(target['name'])
        
        # Combine and extend image list
        all_images.extend(sdss_images + hubble_images)
    
    # Preprocess images
    processed_images = manager.preprocess_fits_images(all_images)
    
    print(f"Total downloaded images: {len(all_images)}")
    print(f"Total processed images: {len(processed_images)}")

if __name__ == '__main__':
    download_reference_datasets()