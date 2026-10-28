import os
import sys
import glob
import h5py
import numpy as np
import pandas as pd
import uproot

def process_file(src_path, dst_path):
    """Processes a single ROOT file, extracts jet and cluster information,

    applies standardization, and dumps structured arrays into an HDF5 file.
    """
    try:
        
        f = uproot.open(src_path)
        tree = f['ClusterTree']
        jet_cols = ['jetRawE', 'jetRawPt', 'jetRawEta', 'nPrimVtx', 'avgMu', 'jetCalE']
        cluster_cols = [
            'clusterE', 'clusterEta', 'cluster_SIGNIFICANCE', 'cluster_time',
            'cluster_SECOND_TIME', 'cluster_CENTER_LAMBDA', 'cluster_CENTER_MAG',
            'cluster_ENG_FRAC_EM_INCL', 'cluster_FIRST_ENG_DENS', 'cluster_LONGITUDINAL',
            'cluster_LATERAL', 'cluster_PTD', 'cluster_ISOLATION'
        ]
        target_col = 'cluster_ENG_CALIB_TOT'
        all_fields = ['jetCnt'] + jet_cols + cluster_cols + [target_col]
 
        df = tree.arrays(all_fields, library='pd')

        all_cluster_fields = cluster_cols + [target_col]
        df_raw = df[['jetCnt'] + all_cluster_fields].copy()
        def scale_log10_standard(series, xmin, epsilon, mean, std):
            x = series.copy()
            if xmin <= 0:
                x = x - xmin + epsilon
            log_x = np.log10(np.maximum(np.abs(x), 1e-10))
            return (log_x - mean) / std

        def scale_standard(series, mean, std):
            return (series - mean) / std

        def scale_max_absolute(series, max_val):
            return series / np.abs(max_val)

        def scale_min_max(series, min_val, max_val):
            return (series - min_val) / (max_val - min_val)

        df['clusterE'] = scale_log10_standard(df['clusterE'], 0, 0, 0.6347346839702431, 0.7385571482973289)
        df['clusterEta'] = scale_standard(df['clusterEta'], 0.010181442396798685, 1.0730789413750401)
        df['cluster_SIGNIFICANCE'] = scale_log10_standard(df['cluster_SIGNIFICANCE'], 0, 0, 0.8866835528437869, 0.5548024471260036)
        df['cluster_time'] = scale_max_absolute(df['cluster_time'], 256.52423095703125)
        df['cluster_SECOND_TIME'] = scale_log10_standard(df['cluster_SECOND_TIME'], 0.0, 1e-10, -0.8385087991536135, 3.5133484230525402)
        df['cluster_CENTER_LAMBDA'] = scale_log10_standard(df['cluster_CENTER_LAMBDA'], 0, 0, 2.454458464238727, 0.455548085569537)
        df['cluster_CENTER_MAG'] = scale_standard(df['cluster_CENTER_MAG'], 2901.331867871424, 1056.6397559178897)
        df['cluster_ENG_FRAC_EM_INCL'] = scale_min_max(df['cluster_ENG_FRAC_EM_INCL'], -2882.9208984375, 15179.9755859375) 
        df['cluster_FIRST_ENG_DENS'] = scale_log10_standard(df['cluster_FIRST_ENG_DENS'], 0, 0, -6.154656057568099, 0.6910824492265635)
        df['cluster_LONGITUDINAL'] = scale_standard(df['cluster_LONGITUDINAL'], 0.7223626116469644, 0.20625686889852246)
        df['cluster_LATERAL'] = scale_standard(df['cluster_LATERAL'], 0.7833833900325896, 0.18259941395116977)
        df['cluster_PTD'] = scale_standard(df['cluster_PTD'], 0.34957693080870894, 0.14286381130028875)
        df['cluster_ISOLATION'] = scale_standard(df['cluster_ISOLATION'], 0.6027900102955646, 0.2512761803141686)
        df['nPrimVtx'] = scale_standard(df['nPrimVtx'], 20.90668211764706, 7.262609335567245)
        df['avgMu'] = scale_standard(df['avgMu'], 34.505480862450824, 12.350157368609814)
        df_jet = df[['jetCnt'] + jet_cols].groupby('jetCnt').first()
        jet_dtype = [(col, 'f4') for col in jet_cols]
        jet_structured = np.array(list(df_jet.to_records(index=False)), dtype=jet_dtype)

        MAX_OBJECTS = 40
        cluster_arr_list = []
        raw_cluster_arr_list = []

        base_dtype = [(field, 'f4') for field in all_cluster_fields]
        target_dtype = base_dtype + [('valid', 'i1')]
        
        grouped = df.groupby('jetCnt').groups
        for jet_id, group_idx in grouped.items():
            elements_to_copy = min(len(group_idx), MAX_OBJECTS)

            arr = df[all_cluster_fields].iloc[group_idx].to_records(index=False)
            padded_arr = np.zeros(MAX_OBJECTS, dtype=target_dtype)
            for field in all_cluster_fields:
                padded_arr[field][:elements_to_copy] = arr[field][:elements_to_copy]
            padded_arr['valid'][:elements_to_copy] = 1
            cluster_arr_list.append(padded_arr)

            arr_raw = df_raw[all_cluster_fields].iloc[group_idx].to_records(index=False)
            padded_raw_arr = np.zeros(MAX_OBJECTS, dtype=target_dtype)
            for field in all_cluster_fields:
                padded_raw_arr[field][:elements_to_copy] = arr_raw[field][:elements_to_copy]
            padded_raw_arr['valid'][:elements_to_copy] = 1
            raw_cluster_arr_list.append(padded_raw_arr)

        clusters_structured = np.array(cluster_arr_list, dtype=target_dtype)
        clusters_raw_structured = np.array(raw_cluster_arr_list, dtype=target_dtype)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        with h5py.File(dst_path, 'w') as hf:
            hf.create_dataset('jet', data=jet_structured)
            hf.create_dataset('clusters', data=clusters_structured)
            hf.create_dataset('clusters_raw', data=clusters_raw_structured)



    except Exception as e:
        

if __name__ == "__main__":
    if len(sys.argv) == 3:
        process_file(sys.argv[1], sys.argv[2])
    else:

        base_src_dir = "/nfs/mahler_2/jhubuserdirs/biswas/RAML_dataset/gnn-hgm/mc20_v1"
        base_dst_dir = "/nfs/mahler_2/jhubuserdirs/gu_parackal/salt/salt/processed_dataset/mc20_v1/train"
        search_pattern = os.path.join(base_src_dir, "**/*.root/*.root")
        root_files = glob.glob(search_pattern, recursive=True)

        skipped_count = 0
        processed_count = 0
        
        for src_f in root_files:
            bname = os.path.basename(src_f).replace(".root", ".h5")
            dst_f = os.path.join(base_dst_dir, bname)
            if os.path.exists(dst_f):
                skipped_count += 1
                continue
                
            process_file(src_f, dst_f)
            processed_count += 1
            
       
