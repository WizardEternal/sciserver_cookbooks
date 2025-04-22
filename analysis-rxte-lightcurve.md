---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.0
  kernelspec:
    display_name: (heasoft)
    language: python
    name: conda-env-heasoft-py
---

# RXTE Lightcurve Exctraction On Sciserver with Parallelization
<hr style="border: 2px solid #fadbac" />

- **Description:** Finding data and extracting light curves from RXTE data, with an example of running tasks in parallel.
- **Level:** Advanced.
- **Data:** RXTE observations of **eta car** taken over 16 years.
- **Requirements:** `heasoftpy`, `pyvo`, `matplotlib`, `tqdm`
- **Credit:** Tess Jaffe (Sep 2021). Parallelization by Abdu Zoghbi (Jan 2024)
- **Support:** Contact the [HEASARC helpdesk](https://heasarc.gsfc.nasa.gov/cgi-bin/Feedback).
- **Last verified to run:** 02/28/2024.

<hr style="border: 2px solid #fadbac" />


## 1. Introduction

This notebook demonstrates an analysis of 16 years of RXTE data, which would be difficult outside of SciServer. 

The RXTE archive contain standard data product that can be used without re-processing the data. These are described in details in the [RXTE ABC guide](https://heasarc.gsfc.nasa.gov/docs/xte/abc/front_page.html).

We first find all of the standard product light curves. Then, realizing that the channel boundaries in the standard data products do not address our science question, we re-extract light curves following the RXTE documentation and using `heasoftpy`.

As we will see, a single run on one observations takes about 20 seconds, which means that extracting all observations takes about a week. We will show an example of how this can be overcome by parallizing the analysis, reducing the run time from weeks to a few hours.

<div style='color: #333; background: #ffffdf; padding:20px; border: 4px solid #fadbac'>
<b>Running On Sciserver:</b><br>
When running this notebook inside Sciserver, make sure the HEASARC data drive is mounted when initializing the Sciserver compute container. <a href='https://heasarc.gsfc.nasa.gov/docs/sciserver/'>See details here</a>.
<br>
Also, this notebook requires <code>heasoftpy</code>, which is available in the (heasoft) conda environment. You should see (heasoft) at the top right of the notebook. If not, click there and select it.

<b>Running Outside Sciserver:</b><br>
If running outside Sciserver, some changes will be needed, including:<br>
&bull; Make sure heasoftpy and heasoft are installed (<a href='https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/'>Download and Install heasoft</a>).<br>
&bull; Unlike on Sciserver, where the data is available locally, you will need to download the data to your machine.<br>
</div>


## 2. Module Imports
We need the following python modules:


```python
import sys, os, shutil
import glob
import pyvo as vo
import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import datetime
import logging

import heasoftpy as hsp

# for prallelization
import multiprocessing as mp

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from itertools import islice
from IPython.display import display

import pandas as pd
import tempfile

# Configure logging for errors
logging.basicConfig(
    filename='rxte_lightcurve.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def simple_progress_bar(total, current, bar_length=30):
    """Custom progress bar to avoid tqdm overhead."""
    percent = current / total
    filled_length = int(bar_length * percent)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    print(f'\r[{bar}] {percent*100:.2f}% ({current}/{total})', end='\r')

# 🗄️ Set up Temp Directory in Memory
os.environ['TMPDIR'] = '/dev/shm'
```

## 3. Find the Data

To find the relevent data, we can use [Xamin](https://heasarc.gsfc.nasa.gov/xamin/), the HEASARC web portal, or the Virtual Observatory (VO) python client `pyvo`. Here, we use the latter so it is all in one notebook.

You can also see the [Getting Started](getting-started.md), [Data Access](data-access.md) and  [Finding and Downloading Data](data-find-download.md) tutorials for examples using `pyVO` to find the data.

Specifically, we want to look at the observation tables.  So first we get a list of all the tables HEASARC serves and then look for the ones related to RXTE:

```python
tap_services = vo.regsearch(servicetype='tap', keywords=['heasarc'])
heasarc_tables = tap_services[0].service.tables
```

```python
for tablename in heasarc_tables.keys():
    if "xte" in tablename:  
        print(" {:20s} {}".format(tablename,heasarc_tables[tablename].description))

```

The `xtemaster` catalog is the one that we are interested in.  

Let's see what this table has in it.  Alternatively, we can google it and find the same information here:

https://heasarc.gsfc.nasa.gov/W3Browse/all/xtemaster.html


```python
for c in heasarc_tables['xtemaster'].columns:
    print("{:20s} {}".format(c.name,c.description))
```

We're interested in Eta Carinae, and we want to get the RXTE `cycle`, `proposal`, and `obsid`. for every observation it took of this source based on its position.  We use the source positoin instead of the name in case the name has been entered differently in the table, which can happen. 

We construct a query in the ADQL language to select the columns (`target_name`, `cycle`, `prnb`, `obsid`, `time`, `exposure`, `ra` and `dec`) where the point defined by the observation's RA and DEC lies inside a circle defined by our chosen source position.

For more example on how to use these powerful VO services, see the [Data Access](data-access.md) and  [Finding and Downloading Data](data-find-download.md) tutorials, or the [NAVO](https://heasarc.gsfc.nasa.gov/vo/summary/python.html) [workshop tutorials](https://nasa-navo.github.io/navo-workshop/).

```python
# Get the coordinate for Eta Car
pos = SkyCoord.from_name("eta car")
query = f"""
SELECT target_name, cycle, prnb, obsid, time, exposure, ra, dec, 
    sqrt(power(cat.ra - {pos.ra.deg}, 2) + power(cat.dec - {pos.dec.deg}, 2)) * 3600 as offset_arcsec
FROM public.xtemaster as cat 
WHERE 
    contains(point('ICRS', cat.ra, cat.dec), circle('ICRS', {pos.ra.deg}, {pos.dec.deg}, 0.1)) = 1 
    AND cat.exposure > 0 
ORDER BY cat.time
"""
```

```python
results=tap_services[0].search(query).to_table()

# Show full results in a DataFrame
results_df = results.to_pandas()
display(results_df)
```

Let's just see how long these observations are:

```python
plt.plot(results['time'],results['exposure'])
plt.xlabel('Time (s)')
plt.ylabel('Exposure (s)')
```

## 4. Read the Standard Products and Plot the Light Curve

Let's collect all the standard product light curves for RXTE.  (These are described on the [RXTE analysis pages](https://heasarc.gsfc.nasa.gov/docs/xte/recipes/cook_book.html).)

```python
## Construct a file list.

def check_file_exists(path):
    """Check if file exists in a multi-threaded pool."""
    return path if os.path.exists(path) else None

rxtedata = "/FTP/rxte/data/archive"
filenames = [
    f"{rxtedata}/AO{cycle}/P{prnb}/{obsid}/stdprod/xp{obsid.replace('-', '')}_n2a.lc.gz"
    for cycle, prnb, obsid in zip(results['cycle'], results['prnb'], results['obsid'])
]

# Check in parallel for missing files
total_files = len(filenames)
valid_files = []

with ThreadPoolExecutor(max_workers=16) as executor:
    for i, file in enumerate(executor.map(check_file_exists, filenames)):
        if file: 
            valid_files.append(file)
        simple_progress_bar(total_files, i + 1)  # Update progress bar

filenames = [file for file in valid_files if file]
print(f"\nFound {len(filenames)} files out of {len(results)} observations.")
```

Let's collect them all into one light curve:

```python
lcurves = []
def load_and_plot_fits_file(file, i, total_files):
    """Load a FITS file and return its lightcurve data."""
    try:
        with fits.open(file) as hdul:
            data = hdul[1].data  # Extract lightcurve data from the second header
        simple_progress_bar(total_files, i + 1)  # Update progress bar
        return data  # Return the lightcurve data
    except Exception as e:
        logging.error(f"Failed to load file {file}: {e}")
        return None

total_files = len(filenames)

with ThreadPoolExecutor(max_workers=16) as executor:
    futures = {executor.submit(load_and_plot_fits_file, file, i, total_files): file for i, file in enumerate(filenames)}
    for i, future in enumerate(futures):
        simple_progress_bar(total_files, i + 1)  # Update progress bar
        result = future.result()
        if result is not None:
            lcurves.append(result)

plt.figure(figsize=(10, 6))
for i, data in enumerate(lcurves):
    plt.plot(data['TIME'], data['RATE'], alpha=0.6)  # Plot each lightcurve with slight transparency
plt.show()
```

```python
# combine the ligh curves into one
def align_and_concatenate_lcurves(lcurves):
    """Align fields in all lightcurves and concatenate them."""
    # Find the union of all field names in the lightcurves
    all_fields = set()
    for lc in lcurves:
        all_fields.update(lc.dtype.names)
    
    all_fields = sorted(all_fields)

    aligned_lcurves = []
    for i, lc in enumerate(lcurves):
        current_fields = lc.dtype.names  # Fields present in the current lightcurve
        new_dtype = [(field, lc.dtype[field]) if field in current_fields else (field, 'f8') for field in all_fields]
        
        # Create an empty structured array with the new dtype
        new_lc = np.zeros(lc.shape, dtype=new_dtype)
        
        for field in lc.dtype.names:
            new_lc[field] = lc[field]
        
        aligned_lcurves.append(new_lc)

    concatenated_lcurve = np.concatenate(aligned_lcurves)

    # The above LCs are merged per proposal.  You can see that some proposals
    # had data added later, after other proposals, so you need to sort:
    concatenated_lcurve.sort(order='TIME') 
    return concatenated_lcurve

lcurve = align_and_concatenate_lcurves(lcurves)

plt.figure(figsize=(10, 6))
plt.plot(lcurve['TIME'], lcurve['RATE'])
plt.show()
```

## 5. Re-extract the Light Curve

Let's say we find that we need different channel boundaries than were used in the standard products.  We can write a function that does the RXTE data analysis steps for every observation to extract a lightcurve and read it into memory to recreate the above dataset.  This function may look complicated, but it only calls three RXTE executables:

* `pcaprepobsid`
* `maketime`
* `pcaextlc2`

which extracts the Standard mode 2 data (not to be confused with the "standard products") for the channels we are interested in.  It has a bit of error checking that'll help when launching a long job.


```python

class XlcError(Exception):
    pass


#  Define a function that, given an ObsID, does the rxte light curve extraction
def rxte_lc(obsid, ao, chmin=5, chmax=10, cleanup=True):
    """Extract RXTE lightcurve for a given ObsID."""
    import tempfile  # Re-import here for multiprocessing
    outdir = tempfile.mkdtemp(prefix=f"tmp.{obsid}.")
    obsdir = f"/FTP/rxte/data/archive/AO{ao}/P{obsid[:5]}/{obsid}/"
    try:
        result = hsp.pcaprepobsid(indir=obsdir, outdir=outdir)
        
        filt_files = glob.glob(f"{outdir}/FP_*.xfl")
        if not filt_files:
            available_files = glob.glob(f"{outdir}/*")
            logging.error(f"No FP_*.xfl file found for ObsID {obsid} in {outdir}. Available files: {available_files}")
            raise XlcError(f"Failed to find FP_*.xfl for ObsID {obsid}")
        
        filt_file = filt_files[0]

        result = hsp.maketime(infile=filt_file, outfile=os.path.join(outdir, 'rxte_example.gti'), 
                              expr="(ELV > 4) && (OFFSET < 0.1)", name='NAME', value='VALUE', time='TIME', compact='NO')

        result = hsp.pcaextlc2(src_infile=f"@{outdir}/FP_dtstd2.lis", bkg_infile=f"@{outdir}/FP_dtbkg2.lis", 
                               outfile=os.path.join(outdir, 'rxte_example.lc'), 
                               gtiandfile=os.path.join(outdir, 'rxte_example.gti'), 
                               chmin=chmin, chmax=chmax, pculist='ALL', layerlist='ALL', binsz=16)

        lc_path = os.path.join(outdir, 'rxte_example.lc')

        if not os.path.exists(lc_path):
            logging.error(f"Lightcurve file {lc_path} not found for ObsID {obsid}")
            raise FileNotFoundError(f"rxte_example.lc not found for ObsID {obsid} in {outdir}")

        with fits.open(lc_path) as hdul:
            lc = hdul[1].data

        return lc

    except Exception as e:
        logging.error(f"Error processing ObsID {obsid}: {e}")
        return None  # Return None if an error occurs

    finally:
        shutil.rmtree(outdir, ignore_errors=True)

```

Note that each call to this function will take 10-20 seconds to complete.  Extracting all the observations will take a while, so we limit this run for 10 observations. We will look into running this in parallel in the next step.

Our new light curves will be for channels 5-10

```python
# For this tutorial, we limit the number of observations to 10
nlimit = 10
total_obs = len(results[:nlimit])

lcurves = []
for i, val in enumerate(results[:nlimit]):
    simple_progress_bar(total_obs, i + 1)  # Update progress bar for observation load
    try:
        lc = rxte_lc(obsid=val['obsid'], ao=val['cycle'], chmin=5, chmax=10, cleanup=True)
        if lc is not None:
            lcurves.append(lc)  # Store only non-None lightcurves
    except Exception as e:
        logging.error(f"Failed to extract lightcurve for ObsID {val['obsid']}: {e}")
        print(f"Failed to extract lightcurve for ObsID {val['obsid']}: {e}")
```

## 6. Running the Extraction in Parallel.
As noted, extracting the light curves for all observations will take a while if run in serial. We will look next into parallizing the `rxte_lc` calls. We will use the `multiprocessing` python module.

We do this by first creating a wrapper around `rxte_lc` that does a few things:

- Use `local_pfiles_context` in `heasoftpy` to properly handle parameter files used by the heasoft tasks. This step is required to prevent parallel calls to `rxte_lc` from reading or writing to the same parameter files that ca lead to calls with the wrong parameters.
- Convert `rxte_lc` from multi-parameter method to a single one, so `multiprocessing` can handle it.
- Catch all errors in the `rxte_lc` call.

We will use all CPUs available in the machine. This can be changing the value of `ncpu`.

```python
def rxte_lc_wrapper(pars):
    """Wrapper for RXTE lightcurve extraction to be used in multiprocessing."""
    obsid, ao, chmin, chmax, cleanup, progress, lock = pars  # Added progress and lock
    try:
        lc = rxte_lc(obsid, ao, chmin, chmax, cleanup)
    except Exception as e:
        logging.error(f"Error in rxte_lc_wrapper for ObsID {obsid}: {e}")
        print(f"Error in rxte_lc_wrapper for ObsID {obsid}: {e}")
        lc = None  # Return None to continue processing
    finally:
        with lock:  # Use a shared lock to ensure thread-safe progress update
            progress.value += 1  # Safely increment the shared counter
    return lc
```

Before running the function in parallel, we construct a list `pars` that holds the parameters that will be passed to `rxte_lc_wrapper` (and hence rxte_lc).

<div style='color: #333; background: #ffffdf; padding:20px; border: 4px solid #fadbac'>
<code>nlimit</code> is now increased to 64. When you run this in full, change the limit to the number of observations
</div>


```python
nlimit = 64
ncpu = mp.cpu_count()
total_obs = len(list(islice(results, nlimit)))  # Total number of observations to be processed
manager = mp.Manager()
progress = manager.Value('i', 0)  # Shared counter to track progress
lock = manager.Lock()  # Lock for thread-safe updates to progress

# Generate parameters for pool
pars = [[val['obsid'], val['cycle'], 5, 10, True, progress, lock] for val in islice(results, nlimit)]  # Pass progress & lock

#  Use imap_unordered to get results as they complete
lcs = []  # Store all successfully extracted lightcurves
with mp.Pool(processes=ncpu) as pool:
    for lc in pool.imap_unordered(rxte_lc_wrapper, pars):  # Using imap_unordered for early results
        if lc is not None:
            lcs.append(lc)  # Only store non-None lightcurves
        
        # Progress bar for shared progress tracking
        simple_progress_bar(total_obs, progress.value)  # Update the progress bar
```

```python
# combine the ligh curves into one while removing None values
lcs = [lc for lc in lcs if lc is not None]

if lcs:
    lcurve = align_and_concatenate_lcurves(lcs)  # Align and combine all lightcurves into one
else:
    logging.error("No lightcurves to combine.")
    lcurve = None

# Plot Final Lightcurve
if lcurve is not None:
    plt.figure(figsize=(8, 4))
    plt.plot(lcurve['TIME'], lcurve['RATE'])
    plt.xlabel('Time (s)')
    plt.ylabel('Rate ($s^{-1}$)')
    plt.title('Combined Lightcurve from All Files')
    plt.show()
else:
    logging.error("No lightcurve to plot.")
```

With the parallelization, we can do more observations at a fraction of the time.

If you want run this notebook on all observations, you can comment out the two cell that runs in serial (the cell below where `rxte_lc` is defined), and submit this notebook in the [batch queue](https://apps.sciserver.org/compute/jobs) on Sciserver.

