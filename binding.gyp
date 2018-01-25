{
  "targets": [
    {
      "target_name": "nen",
      "sources": [ "nen_node.cpp" ],
      'cflags': [
        '-fopenmp',
      ],
      'ldflags': [
        '-fopenmp',
      ],
      'conditions': [
        ['OS=="win"', {
          'msvs_settings': {
            'VCCLCompilerTool': {
              'AdditionalOptions': [
                '/openmp'
              ]
            }
          }
		}]
	  ]
	}
  ]
}