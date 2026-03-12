import os
import urllib.request
import urllib.error
from astroquery.cadc import Cadc

def download_cadc_dark():
    cadc = Cadc()
    os.makedirs("benchmark_data", exist_ok=True)
    
    print("Querying CADC for public CFHT MegaCam DARK exposures...")
    
    query = """
    SELECT TOP 10
        Plane.publisherID as publisherID, Observation.observationID
    FROM caom2.Plane AS Plane 
    JOIN caom2.Observation AS Observation ON Plane.obsID = Observation.obsID 
    WHERE Observation.collection = 'CFHT' 
      AND Observation.instrument_name = 'MegaPrime' 
      AND Observation.type = 'DARK' 
      AND Plane.dataRelease < '2010-01-01'
    """
    
    try:
        results = cadc.exec_sync(query)
        if len(results) > 0:
            print(f"Resolving URLs for {len(results)} rows...")
            urls = cadc.get_data_urls(results)
            print(f"Found {len(urls)} URLs")
            for url in urls:
                pub_id = url.split('/')[-1]
                out_file = os.path.join("benchmark_data", f"cfht_dark_{pub_id}.fits")
                print(f"Downloading from {url} ...")
                try:
                    urllib.request.urlretrieve(url, out_file)
                    print(f"Saved to {out_file}")
                    return
                except Exception as e:
                    print(f"  Download failed: {e}")
    except Exception as e:
        print(f"Query or Resolution failed: {e}")

if __name__ == "__main__":
    download_cadc_dark()
