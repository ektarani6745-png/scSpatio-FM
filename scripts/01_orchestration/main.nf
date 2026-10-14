nextflow.enable.dsl=2

params.dataset_url = "https://cf.10xgenomics.com/samples/spatial-exp/1.0.0/V1_Human_Lymph_Node/V1_Human_Lymph_Node_filtered_feature_bc_matrix.h5"
params.image_url   = "https://cf.10xgenomics.com/samples/spatial-exp/1.0.0/V1_Human_Lymph_Node/V1_Human_Lymph_Node_spatial.tar.gz"

process DOWNLOAD_SPATIAL_DATA {
    publishDir "../../data/raw", mode: 'copy'

    output:
    path "V1_Human_Lymph_Node_filtered_feature_bc_matrix.h5", emit: h5_matrix
    path "V1_Human_Lymph_Node_spatial.tar.gz", emit: spatial_tar

    script:
    """
    curl -L -O ${params.dataset_url}
    curl -L -O ${params.image_url}
    """
}

process EXTRACT_IMAGES {
    publishDir "../../data/raw", mode: 'copy'

    input:
    path spatial_tar

    output:
    path "spatial"

    script:
    """
    tar -xzvf ${spatial_tar}
    """
}

workflow {
    DOWNLOAD_SPATIAL_DATA()
    EXTRACT_IMAGES(DOWNLOAD_SPATIAL_DATA.out.spatial_tar)
}
