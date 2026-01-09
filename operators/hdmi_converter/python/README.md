### QCAP

The `qcap_source` extension supports YUAN High-Tech capture cards as
the video source.

#### `nvidia::holoscan::QCAPSource`

QCAP Source codelet

##### Parameters

- **`allocator`**: allocator (default: `null`)
  - type: `holoscan::Allocator`
- **`cuda_device_ordinal`**: cuda device ordinal (default: `0`)
  - type: `int`
- **`out_tensor_name`**: tensor name of ouput (default: ` `)
  - type: `str`
- **`left_tensor_name`**: tensor name of left eye (default: ` `)
  - type: `str`
- **`right_tensor_name`**: tensor name of right eye (default: ` `)
  - type: `str`
