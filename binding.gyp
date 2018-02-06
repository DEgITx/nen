{
  "targets": [
    {
      "target_name": "nen",
      "sources": [ "src/nen_node.cpp" ],
      'cflags': [
        '-fopenmp',
      ],
      'ldflags': [
        '-fopenmp',
      ],
      'xcode_settings': {
        'OTHER_CFLAGS': ['-fopenmp'],
        'OTHER_LDFLAGS': ['-lomp'],
        'CLANG_CXX_LANGUAGE_STANDART': 'c++11'
      },
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