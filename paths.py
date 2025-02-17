# defines file paths for any given task

def get_paths(task):
	paths = {
		'treecover': {
			'home_dir': 'data_aug/',
			'img_dir': 'data_aug/data/mosaiks/png/contus_uar/',
			'log_dir': 'data_aug/outputs/treecover/runs/',
			'model_dir': 'data_aug/outputs/treecover/models/',
			'means': 'data_aug/channel_means/channel_means_contus_uar_Apr-06-2022.txt',
		},

		'nightlights': {
			'home_dir': 'data_aug/',
			'img_dir': 'data_aug/data/mosaiks/png/contus_pop/',
			'log_dir': 'data_aug/outputs/nightlights/runs/',
			'model_dir': 'data_aug/outputs/nightlights/models/',
			'means': 'data_aug/channel_means/channel_means_contus_pop_Nov-16-2022.txt',
		},

		'elevation': {
			'home_dir': 'data_aug/',
			'img_dir': 'data_aug/data/mosaiks/png/contus_uar/',
			'log_dir': 'data_aug/outputs/elevation/runs/',
			'model_dir': 'data_aug/outputs/elevation/models/',
			'means': 'data_aug/channel_means/channel_means_contus_uar_Apr-06-2022.txt',
		},

		'population': {
			'home_dir': 'data_aug/',
			'img_dir': 'data_aug/data/mosaiks/png/contus_pop/',
			'log_dir': 'data_aug/outputs/population/runs/',
			'model_dir': 'data_aug/outputs/population/models/',
			'means': 'data_aug/channel_means/channel_means_contus_pop_Nov-16-2022.txt',
		},

		'landuse': {
			'home_dir': 'data_aug/',
			'img_dir': 'data_aug/data/ucMerced_landuse/data/npy/',
			'log_dir': 'data_aug/outputs/ucMerced_landuse/runs/',
			'model_dir': 'data_aug/outputs/ucMerced_landuse/models/',
			'means': 'data_aug/channel_means/channel_means_ucMerced_landuse_Jan-05-2023.txt',
		},

		'coffee': {
			'home_dir': 'data_aug/',
			'img_dir': 'data_aug/data/coffee/data/jpg/',
			'log_dir': 'data_aug/outputs/coffee/runs/',
			'model_dir': 'data_aug/outputs/coffee/models/',
			'means': 'data_aug/channel_means/channel_means_coffee_Aug-18-2024.txt',
		},

		'eurosat': {
			'home_dir': 'data_aug/',
			'img_dir': 'data_aug/data/eurosat/npy/',
			'log_dir': 'data_aug/outputs/eurosat/runs/',
			'model_dir': 'data_aug/outputs/eurosat/models/',
			'means': 'data_aug/channel_means/channel_means_eurosat_ms_Jul-01-2024.txt',
		}
	}

	return paths[task]

