# FR-SLAM LiDAR Adapter Layer

> **Stage 1 — Multi-LiDAR Input Unification**
>
> This document summarizes the first development stage of **FR-SLAM**: converting heterogeneous LiDAR messages into one internal data format that can be shared by preprocessing, deskewing, registration, odometry, and mapping modules.

---

## Table of Contents

- [1. Goal](#1-goal)
- [2. System Architecture](#2-system-architecture)
- [3. Unified Point Type](#3-unified-point-type)
- [4. Unified LiDAR Frame](#4-unified-lidar-frame)
- [5. Adapter Interface](#5-adapter-interface)
- [6. Livox MID360S Adapter](#6-livox-mid360s-adapter)
- [7. MID360S Tag Handling](#7-mid360s-tag-handling)
- [8. Hesai Adapter](#8-hesai-adapter)
- [9. Unified Time Convention](#9-unified-time-convention)
- [10. ROS2 and PCL Conversion](#10-ros2-and-pcl-conversion)
- [11. ROS Header: `frame_id` and `stamp`](#11-ros-header-frame_id-and-stamp)
- [12. Validation Results](#12-validation-results)
- [13. Design Rules](#13-design-rules)
- [14. Stage 1 Checklist](#14-stage-1-checklist)
- [15. Next Stage](#15-next-stage)

---

# 1. Goal

FR-SLAM currently supports two LiDAR sources:

- **Livox MID360S**
- **Hesai LiDAR**

Their raw point formats are different.

### Livox MID360S

```text
x
y
z
intensity
tag
line
timestamp
```

### Hesai

```text
x
y
z
intensity
ring
timestamp
```

The goal of the adapter layer is:

```text
Sensor-specific ROS2 PointCloud2
              ↓
        Sensor Adapter
              ↓
       Unified LIDAR_FRAME
              ↓
   Common SLAM Processing Pipeline
```

After this layer, the rest of FR-SLAM should not need to know whether the point cloud came from Livox or Hesai.

---

# 2. System Architecture

```text
                        ROS2
                         │
              sensor_msgs::msg::PointCloud2
                         │
              ┌──────────┴──────────┐
              │                     │
         Livox MID360S           Hesai LiDAR
              │                     │
              ▼                     ▼
         LIVOX_POINT            HESAI_POINT
              │                     │
              ▼                     ▼
      Mid360s_Adapter         HESAI_Adapter
              │                     │
      ┌───────┴────────┐            │
      │ tag validation │            │
      │ line → ring    │            │
      │ ns → second    │            │
      └───────┬────────┘            │
              │              timestamp already
              │                  in seconds
              │                     │
              └──────────┬──────────┘
                         ▼
                    LIDAR_FRAME
                         │
              pcl::PointCloud<LIDAR_POINT>
                         │
          x y z intensity ring time_offset
```

The most important design principle is:

> **Sensor differences should end inside the Adapter layer.**

---

# 3. Unified Point Type

FR-SLAM uses one common point representation:

```cpp
struct EIGEN_ALIGN16 LIDAR_POINT
{
    PCL_ADD_POINT4D;

    float intensity;
    std::uint16_t ring;
    double time_offset;

    PCL_MAKE_ALIGNED_OPERATOR_NEW
};
```

The unified fields are:

| Field | Meaning | Unit |
|---|---|---|
| `x` | Cartesian X coordinate | m |
| `y` | Cartesian Y coordinate | m |
| `z` | Cartesian Z coordinate | m |
| `intensity` | Return intensity | sensor-dependent |
| `ring` | Laser / scan channel index | integer |
| `time_offset` | Point acquisition time relative to scan start | s |

The key time relationship is:

$$
t_i = t_{\text{scan start}} + \Delta t_i
$$

where:

```text
t_scan_start = scan_start_time
Δt_i         = point.time_offset
```

This convention will later be used by the IMU deskew module.

---

# 4. Unified LiDAR Frame

One scan is represented by `LIDAR_FRAME`.

Conceptually:

```text
LIDAR_FRAME
├── cloud
├── scan_start_time
├── has_point_time
└── frame_id
```

A typical definition is:

```cpp
struct LIDAR_FRAME
{
    pcl::PointCloud<LIDAR_POINT>::Ptr cloud;

    double scan_start_time = 0.0;

    bool has_point_time = false;

    std::string frame_id;

    LIDAR_FRAME()
        : cloud(pcl::make_shared<pcl::PointCloud<LIDAR_POINT>>())
    {
    }
};
```

## Field meanings

### `cloud`

```cpp
pcl::PointCloud<LIDAR_POINT>::Ptr cloud;
```

Stores the unified points of the current scan.

### `scan_start_time`

```cpp
double scan_start_time;
```

Absolute timestamp of the scan reference start time.

Unit:

```text
second
```

### `has_point_time`

```cpp
bool has_point_time;
```

Indicates whether the current scan contains meaningful per-point timing information.

Example:

```text
point 0 → 0.000 s
point 1 → 0.004 s
point 2 → 0.009 s
...
```

Then:

```cpp
has_point_time = true;
```

If all points have exactly the same timestamp, per-point timing is not useful:

```cpp
has_point_time = false;
```

### `frame_id`

```cpp
std::string frame_id;
```

The coordinate frame in which the point coordinates are expressed.

Examples:

```text
livox_frame
hesai_lidar
```

---

# 5. Adapter Interface

A common base interface is used for LiDAR conversion:

```cpp
class Lidar_Adapt
{
public:
    virtual ~Lidar_Adapt() = default;

    virtual LIDAR_FRAME convert(
        const sensor_msgs::msg::PointCloud2 &msg) = 0;
};
```

This allows different sensors to provide their own conversion implementation while exposing the same interface.

For example:

```text
Lidar_Adapt
├── Mid360s_Adapter
└── HESAI_Adapter
```

## Why use a virtual destructor?

```cpp
virtual ~Lidar_Adapt() = default;
```

The base class may later be used through a base pointer:

```cpp
std::unique_ptr<Lidar_Adapt>
```

while the real object is:

```text
Mid360s_Adapter
```

or:

```text
HESAI_Adapter
```

A virtual destructor guarantees that derived-class destruction is handled correctly.

## What does `= default` mean?

```cpp
= default
```

asks the compiler to generate the default implementation automatically.

---

# 6. Livox MID360S Adapter

The MID360S raw format contains:

```text
x
y
z
intensity
tag
line
timestamp
```

The adapter performs:

```text
x          → x
y          → y
z          → z
intensity  → intensity
line       → ring
timestamp  → scan_start_time + time_offset
tag        → validated inside the adapter
```

## 6.1 Raw cloud conversion

```cpp
pcl::PointCloud<LIVOX_POINT>::Ptr livox_cloud =
    pcl::make_shared<pcl::PointCloud<LIVOX_POINT>>();

pcl::fromROSMsg(msg, *livox_cloud);
```

## 6.2 Empty cloud check

```cpp
if (livox_cloud->empty())
{
    lidar_frame.has_point_time = false;
    return lidar_frame;
}
```

Using an early return keeps the rest of the conversion logic simple.

## 6.3 Find the scan time range

```cpp
double min_timestamp =
    std::numeric_limits<double>::max();

double max_timestamp =
    std::numeric_limits<double>::lowest();
```

Then scan all points:

```cpp
for (const LIVOX_POINT &point : livox_cloud->points)
{
    if (point.timestamp < min_timestamp)
    {
        min_timestamp = point.timestamp;
    }

    if (point.timestamp > max_timestamp)
    {
        max_timestamp = point.timestamp;
    }
}
```

### Important note about `numeric_limits`

```cpp
std::numeric_limits<double>::max()
```

is approximately:

```text
+1.797e308
```

and is useful when searching for a minimum value.

```cpp
std::numeric_limits<double>::lowest()
```

is approximately:

```text
-1.797e308
```

and is useful when searching for a maximum value.

Do not confuse:

```cpp
std::numeric_limits<double>::min()
```

with the most negative floating-point value. For floating-point types, `min()` means the smallest positive normal value.

---

# 7. MID360S Tag Handling

`tag` is a Livox-specific field.

It should be interpreted inside `Mid360s_Adapter` and should **not** be added to the common `LIDAR_POINT`.

The architecture is therefore:

```text
LIVOX_POINT
    │
    ├── validate tag
    │
    ├── convert line → ring
    │
    └── convert timestamp
    │
    ▼
LIDAR_POINT
```

## 7.1 Extracting bit groups

Example:

```cpp
const std::uint8_t other_status =
    (tag >> 4) & 0x03;

const std::uint8_t rain_status =
    (tag >> 2) & 0x03;

const std::uint8_t glue_status =
    tag & 0x03;
```

`0x03` is:

```text
00000011
```

so:

```cpp
& 0x03
```

keeps only the lowest two bits.

For example:

```text
tag = 00110110
```

can be separated into:

```text
00 | 11 | 01 | 10
```

The three extracted two-bit values are:

```text
bit 5~4 → 3
bit 3~2 → 1
bit 1~0 → 2
```

## 7.2 Validation helper

The tag logic is kept outside `convert()`:

```cpp
bool isValidTag(std::uint8_t tag) const;
```

Then the conversion loop stays simple:

```cpp
if (!isValidTag(raw_point.tag))
{
    continue;
}
```

This means:

```text
invalid tag
    ↓
skip current point
    ↓
continue with the next point
```

## 7.3 Why is the function `const`?

```cpp
bool isValidTag(std::uint8_t tag) const;
```

The final `const` means that the function does not modify the state of the current `Mid360s_Adapter` object.

---

# 8. Hesai Adapter

The Hesai raw point format is:

```text
x
y
z
intensity
ring
timestamp
```

The conversion is simpler:

```text
x          → x
y          → y
z          → z
intensity  → intensity
ring       → ring
timestamp  → scan_start_time + time_offset
```

The raw type should only exist on the sensor-specific side:

```text
HESAI_POINT
    ↓
HESAI_Adapter
    ↓
LIDAR_POINT
```

Do not push a `HESAI_POINT` directly into:

```cpp
pcl::PointCloud<LIDAR_POINT>
```

Instead create a unified point:

```cpp
LIDAR_POINT lidar_point;

lidar_point.x = raw_point.x;
lidar_point.y = raw_point.y;
lidar_point.z = raw_point.z;
lidar_point.intensity = raw_point.intensity;
lidar_point.ring = raw_point.ring;
```

and then:

```cpp
lidar_frame.cloud->push_back(lidar_point);
```

---

# 9. Unified Time Convention

The two sensors use different raw timestamp scales.

## 9.1 MID360S

MID360S raw timestamps are nanosecond-scale values.

Therefore:

```cpp
lidar_frame.scan_start_time =
    min_timestamp * 1e-9;
```

and:

```cpp
lidar_point.time_offset =
    (raw_point.timestamp - min_timestamp) * 1e-9;
```

Both outputs are in seconds.

---

## 9.2 Hesai

The tested Hesai timestamp values are already in seconds.

Therefore:

```cpp
lidar_frame.scan_start_time =
    min_timestamp;
```

and:

```cpp
lidar_point.time_offset =
    raw_point.timestamp - min_timestamp;
```

No additional `1e-9` conversion is applied.

---

## 9.3 Final internal convention

Regardless of sensor:

```text
scan_start_time → second
time_offset     → second
```

Therefore:

$$
t_i =
scan\_start\_time
+
time\_offset_i
$$

This gives a clean interface for future LiDAR-IMU synchronization and motion compensation.

---

# 10. ROS2 and PCL Conversion

The input message is:

```cpp
sensor_msgs::msg::PointCloud2
```

## ROS2 → PCL

```cpp
pcl::fromROSMsg(msg, cloud);
```

Conceptually:

```text
ROS PointCloud2
      ↓
PCL PointCloud
```

## PCL → ROS2

```cpp
pcl::toROSMsg(*lidar_frame.cloud, pub_data);
```

Conceptually:

```text
PCL PointCloud
      ↓
ROS PointCloud2
```

This is useful when publishing the unified point cloud for RViz inspection.

---

# 11. ROS Header: `frame_id` and `stamp`

A `PointCloud2` message contains more than point coordinates.

```text
PointCloud2
│
├── header
│   ├── frame_id
│   └── stamp
│
├── fields
├── width
├── height
└── data
```

## `frame_id`

```cpp
header.frame_id
```

answers:

> **Where is this point cloud expressed?**

Examples:

```text
livox_frame
hesai_lidar
```

## `stamp`

```cpp
header.stamp
```

answers:

> **When does this message belong to?**

A useful mental model is:

```text
frame_id → Where?
stamp    → When?
```

RViz and TF use both pieces of information to locate a point cloud in the correct frame at the correct time.

## Re-publishing an adapted cloud

For adapter validation, the simplest approach is:

```cpp
sensor_msgs::msg::PointCloud2 pub_data;

pcl::toROSMsg(
    *lidar_frame.cloud,
    pub_data);

pub_data.header = data.header;

publisher->publish(pub_data);
```

This preserves the original ROS coordinate frame and message timestamp.

---

# 12. Validation Results

## 12.1 MID360S

Observed example:

```text
ROS raw cloud size: 19968
Unified cloud size: 19917

Frame ID: livox_frame

Has point time: true

Time offset:
0.000000000 ~ 0.100486656 s

Scan duration:
0.100486656 s

Ring Range:
0 ~ 3
```

Interpretation:

- PointCloud2 parsing works.
- Raw Livox points are correctly converted to `LIDAR_POINT`.
- `line → ring` works.
- Per-point timing is valid.
- The scan duration is approximately `0.1 s`.
- Tag filtering is active.
- The unified output contains fewer points than the raw input because rejected tag values are skipped.

For the example above:

```text
19968 - 19917 = 51
```

so 51 points were rejected by tag validation.

---

# 13. Design Rules

The Adapter layer should handle **sensor-specific differences**.

Examples:

```text
MID360S tag
MID360S line
MID360S timestamp unit
Hesai timestamp unit
Hesai-specific point fields
```

The common preprocessing layer should handle **sensor-independent geometry processing**.

Examples:

```text
NaN / Inf removal
zero-point removal
range filtering
ROI filtering
VoxelGrid downsampling
```

The intended boundary is:

```text
Sensor-specific problem
        ↓
      Adapter

Sensor-independent problem
        ↓
   Preprocessor
```

This prevents sensor-specific logic from leaking into the SLAM front-end.

---

# 14. Stage 1 Checklist

## Core Data Types

- [x] Unified `LIDAR_POINT`
- [x] Unified `LIDAR_FRAME`
- [x] Common `Lidar_Adapt` interface

## C++ Interface Design

- [x] Inheritance
- [x] `virtual`
- [x] Virtual destructor
- [x] `override`
- [x] `const` member function

## MID360S

- [x] `PointCloud2 → LIVOX_POINT`
- [x] `line → ring`
- [x] nanoseconds → seconds
- [x] `scan_start_time`
- [x] `time_offset`
- [x] tag bit extraction
- [x] tag validation
- [x] runtime validation

## Hesai

- [x] `PointCloud2 → HESAI_POINT`
- [x] `ring → ring`
- [x] timestamp conversion logic
- [x] `HESAI_POINT → LIDAR_POINT`
- [x] unified frame output

## ROS2 / PCL

- [x] `pcl::fromROSMsg`
- [x] `pcl::toROSMsg`
- [x] `header.frame_id`
- [x] `header.stamp`
- [x] RViz publication test

---

# 15. Next Stage

The next development stage is the **Common LiDAR Preprocessing Layer**.

Input:

```cpp
LIDAR_FRAME
```

Planned pipeline:

```text
LIDAR_FRAME
     ↓
Finite / invalid point check
     ↓
Zero-point removal
     ↓
Range filtering
     ↓
Optional ROI filtering
     ↓
VoxelGrid downsampling
     ↓
Registration input
```

Suggested files:

```text
include/fr_slam/fr_lidar_preprocess.hpp
src/fr_lidar_preprocess.cpp
```

Future pipeline:

```text
LiDAR Driver
    ↓
Adapter
    ↓
Unified LIDAR_FRAME
    ↓
Preprocess
    ↓
Deskew
    ↓
Registration
    ↓
Odometry
    ↓
Local Map
    ↓
SLAM
```

---

## Key Takeaway

> **The Adapter layer is the boundary where sensor differences disappear.**

From `LIDAR_FRAME` onward, FR-SLAM should treat MID360S, Hesai, and future LiDAR sensors through the same processing interface.
