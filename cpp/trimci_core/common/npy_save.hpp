#pragma once
/**
 * npy_save.hpp - Lightweight NPY file writer for C++
 * 
 * Writes NumPy .npy format files directly from C++ without Python dependency.
 * Supports 1D and 2D arrays of common types (double, float, int64, uint64).
 * 
 * NPY format specification: https://numpy.org/devdocs/reference/generated/numpy.lib.format.html
 * 
 * Usage:
 *   std::vector<double> data = {1.0, 2.0, 3.0};
 *   npy::save("output.npy", data);  // 1D array
 *   
 *   std::vector<std::vector<double>> data2d = {{1,2}, {3,4}};
 *   npy::save_2d("output2d.npy", data2d);  // 2D array
 */

#include <vector>
#include <string>
#include <fstream>
#include <cstdint>
#include <stdexcept>
#include <cstring>
#include <array>
#include <type_traits>

namespace npy {

// Type trait for NPY dtype string
template<typename T> struct dtype_str { static constexpr const char* value = nullptr; };
template<> struct dtype_str<double>   { static constexpr const char* value = "<f8"; };
template<> struct dtype_str<float>    { static constexpr const char* value = "<f4"; };
template<> struct dtype_str<int64_t>  { static constexpr const char* value = "<i8"; };
template<> struct dtype_str<int32_t>  { static constexpr const char* value = "<i4"; };
template<> struct dtype_str<uint64_t> { static constexpr const char* value = "<u8"; };
template<> struct dtype_str<uint32_t> { static constexpr const char* value = "<u4"; };

/**
 * Write NPY header
 */
inline void write_header(std::ofstream& f, const std::string& dtype, 
                         const std::vector<size_t>& shape, bool fortran_order = false) {
    // Magic string + version
    const char magic[] = "\x93NUMPY";
    f.write(magic, 6);
    
    // Version 1.0
    char version[] = {0x01, 0x00};
    f.write(version, 2);
    
    // Build header dict
    std::string header = "{'descr': '" + dtype + "', 'fortran_order': ";
    header += (fortran_order ? "True" : "False");
    header += ", 'shape': (";
    for (size_t i = 0; i < shape.size(); ++i) {
        header += std::to_string(shape[i]);
        if (i < shape.size() - 1) header += ", ";
        else if (shape.size() == 1) header += ",";  // Trailing comma for 1D
    }
    header += "), }";
    
    // Pad to 64-byte alignment (total header size = 10 + header_len must be divisible by 64)
    size_t padding_needed = 64 - ((10 + header.size() + 1) % 64);  // +1 for newline
    if (padding_needed == 64) padding_needed = 0;
    header += std::string(padding_needed, ' ');
    header += '\n';
    
    // Write header length (2 bytes, little-endian)
    uint16_t header_len = static_cast<uint16_t>(header.size());
    f.write(reinterpret_cast<const char*>(&header_len), 2);
    
    // Write header
    f.write(header.c_str(), header.size());
}

/**
 * Save 1D array to NPY file
 */
template<typename T>
void save(const std::string& filename, const std::vector<T>& data) {
    static_assert(dtype_str<T>::value != nullptr, "Unsupported type for NPY save");
    
    std::ofstream f(filename, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }
    
    std::vector<size_t> shape = {data.size()};
    write_header(f, dtype_str<T>::value, shape);
    
    // Write data
    f.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(T));
}

/**
 * Save 2D array to NPY file (row-major / C order)
 */
template<typename T>
void save_2d(const std::string& filename, const std::vector<std::vector<T>>& data) {
    static_assert(dtype_str<T>::value != nullptr, "Unsupported type for NPY save");
    
    if (data.empty()) {
        throw std::runtime_error("Cannot save empty 2D array");
    }
    
    std::ofstream f(filename, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }
    
    size_t nrows = data.size();
    size_t ncols = data[0].size();
    
    std::vector<size_t> shape = {nrows, ncols};
    write_header(f, dtype_str<T>::value, shape);
    
    // Write data row by row
    for (const auto& row : data) {
        if (row.size() != ncols) {
            throw std::runtime_error("Inconsistent row sizes in 2D array");
        }
        f.write(reinterpret_cast<const char*>(row.data()), ncols * sizeof(T));
    }
}

/**
 * Save 2D array from flat vector with given shape
 */
template<typename T>
void save_2d_flat(const std::string& filename, const std::vector<T>& data, 
                  size_t nrows, size_t ncols) {
    static_assert(dtype_str<T>::value != nullptr, "Unsupported type for NPY save");
    
    if (data.size() != nrows * ncols) {
        throw std::runtime_error("Data size does not match shape");
    }
    
    std::ofstream f(filename, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }
    
    std::vector<size_t> shape = {nrows, ncols};
    write_header(f, dtype_str<T>::value, shape);
    
    // Write data
    f.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(T));
}

/**
 * Save determinants as 2D uint64 array
 * Each determinant is saved as [alpha_bits, beta_bits] for 64-bit determinants
 */
template<typename DetType>
void save_determinants(const std::string& filename, const std::vector<DetType>& dets) {
    std::ofstream f(filename, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }
    
    size_t nrows = dets.size();
    size_t ncols = 2;  // alpha, beta
    
    std::vector<size_t> shape = {nrows, ncols};
    write_header(f, "<u8", shape);  // uint64
    
    // Write data
    for (const auto& det : dets) {
        uint64_t alpha = det.alpha;
        uint64_t beta = det.beta;
        f.write(reinterpret_cast<const char*>(&alpha), sizeof(uint64_t));
        f.write(reinterpret_cast<const char*>(&beta), sizeof(uint64_t));
    }
}

// ============================================================================
// NPZ Support (ZIP STORED - uncompressed)
// ============================================================================

/**
 * Build NPY data in memory
 */
template<typename T>
std::vector<char> build_npy_data(const std::vector<T>& data, const std::vector<size_t>& shape) {
    std::vector<char> result;
    
    // Magic string + version
    const char magic[] = "\x93NUMPY";
    result.insert(result.end(), magic, magic + 6);
    
    // Version 1.0
    result.push_back(0x01);
    result.push_back(0x00);
    
    // Build header dict
    std::string header = "{'descr': '";
    header += dtype_str<T>::value;
    header += "', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < shape.size(); ++i) {
        header += std::to_string(shape[i]);
        if (i < shape.size() - 1) header += ", ";
        else if (shape.size() == 1) header += ",";
    }
    header += "), }";
    
    // Pad to 64-byte alignment
    size_t padding_needed = 64 - ((10 + header.size() + 1) % 64);
    if (padding_needed == 64) padding_needed = 0;
    header += std::string(padding_needed, ' ');
    header += '\n';
    
    // Write header length
    uint16_t header_len = static_cast<uint16_t>(header.size());
    result.push_back(static_cast<char>(header_len & 0xFF));
    result.push_back(static_cast<char>((header_len >> 8) & 0xFF));
    
    // Write header
    result.insert(result.end(), header.begin(), header.end());
    
    // Write data
    const char* data_ptr = reinterpret_cast<const char*>(data.data());
    result.insert(result.end(), data_ptr, data_ptr + data.size() * sizeof(T));
    
    return result;
}

/**
 * Build NPY data for determinants
 * Supports both 64-bit (uint64_t) and scalable (array<uint64_t, N>) determinants
 */

// Helper trait to detect array type
template<typename T>
struct is_array : std::false_type {};

template<typename T, size_t N>
struct is_array<std::array<T, N>> : std::true_type {};

// Helper to get array size
template<typename T>
struct array_size { static constexpr size_t value = 1; };

template<typename T, size_t N>
struct array_size<std::array<T, N>> { static constexpr size_t value = N; };

template<typename DetType>
std::vector<char> build_npy_dets(const std::vector<DetType>& dets) {
    std::vector<char> result;
    
    using AlphaType = decltype(std::declval<DetType>().alpha);
    constexpr bool is_array_type = is_array<AlphaType>::value;
    constexpr size_t n_segments = array_size<AlphaType>::value;
    
    size_t nrows = dets.size();
    size_t ncols = 2 * n_segments;  // For array types, need more columns
    
    // Magic + version
    const char magic[] = "\x93NUMPY";
    result.insert(result.end(), magic, magic + 6);
    result.push_back(0x01);
    result.push_back(0x00);
    
    // Header
    std::string header = "{'descr': '<u8', 'fortran_order': False, 'shape': (";
    header += std::to_string(nrows) + ", " + std::to_string(ncols) + "), }";
    
    size_t padding_needed = 64 - ((10 + header.size() + 1) % 64);
    if (padding_needed == 64) padding_needed = 0;
    header += std::string(padding_needed, ' ');
    header += '\n';
    
    uint16_t header_len = static_cast<uint16_t>(header.size());
    result.push_back(static_cast<char>(header_len & 0xFF));
    result.push_back(static_cast<char>((header_len >> 8) & 0xFF));
    result.insert(result.end(), header.begin(), header.end());
    
    // Data
    for (const auto& det : dets) {
        if constexpr (is_array_type) {
            // Array-based determinant
            for (size_t i = 0; i < n_segments; ++i) {
                uint64_t val = det.alpha[i];
                result.insert(result.end(), reinterpret_cast<char*>(&val), 
                              reinterpret_cast<char*>(&val) + sizeof(uint64_t));
            }
            for (size_t i = 0; i < n_segments; ++i) {
                uint64_t val = det.beta[i];
                result.insert(result.end(), reinterpret_cast<char*>(&val), 
                              reinterpret_cast<char*>(&val) + sizeof(uint64_t));
            }
        } else {
            // 64-bit determinant
            uint64_t alpha = det.alpha;
            uint64_t beta = det.beta;
            result.insert(result.end(), reinterpret_cast<char*>(&alpha), 
                          reinterpret_cast<char*>(&alpha) + sizeof(uint64_t));
            result.insert(result.end(), reinterpret_cast<char*>(&beta), 
                          reinterpret_cast<char*>(&beta) + sizeof(uint64_t));
        }
    }
    
    return result;
}

/**
 * Write uint32 little-endian
 */
inline void write_u32_le(std::ofstream& f, uint32_t v) {
    char buf[4];
    buf[0] = v & 0xFF;
    buf[1] = (v >> 8) & 0xFF;
    buf[2] = (v >> 16) & 0xFF;
    buf[3] = (v >> 24) & 0xFF;
    f.write(buf, 4);
}

/**
 * Write uint16 little-endian
 */
inline void write_u16_le(std::ofstream& f, uint16_t v) {
    char buf[2];
    buf[0] = v & 0xFF;
    buf[1] = (v >> 8) & 0xFF;
    f.write(buf, 2);
}

/**
 * Save NPZ file with determinants and coefficients
 * Compatible with np.load() 
 */
template<typename DetType>
void save_npz(const std::string& filename,
              const std::vector<DetType>& dets,
              const std::vector<double>& coeffs,
              const std::vector<DetType>* pool = nullptr) {
    
    std::ofstream f(filename, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }
    
    // Build NPY data for each array
    std::vector<std::pair<std::string, std::vector<char>>> arrays;
    
    // dets.npy
    arrays.push_back({"dets.npy", build_npy_dets(dets)});
    
    // core_set.npy (alias)
    arrays.push_back({"core_set.npy", build_npy_dets(dets)});
    
    // dets_coeffs.npy
    arrays.push_back({"dets_coeffs.npy", build_npy_data(coeffs, {coeffs.size()})});
    
    // core_set_coeffs.npy (alias)
    arrays.push_back({"core_set_coeffs.npy", build_npy_data(coeffs, {coeffs.size()})});
    
    // coeffs.npy (alias)
    arrays.push_back({"coeffs.npy", build_npy_data(coeffs, {coeffs.size()})});
    
    // pool.npy (optional)
    if (pool && !pool->empty()) {
        arrays.push_back({"pool.npy", build_npy_dets(*pool)});
    }
    
    // Write ZIP file (STORED method - no compression)
    std::vector<std::tuple<std::string, uint32_t, uint32_t>> central_dir_entries;  // name, offset, size
    
    for (const auto& [name, data] : arrays) {
        uint32_t offset = static_cast<uint32_t>(f.tellp());
        uint32_t data_size = static_cast<uint32_t>(data.size());
        
        // Local file header
        write_u32_le(f, 0x04034b50);  // Local file header signature
        write_u16_le(f, 20);           // Version needed to extract (2.0)
        write_u16_le(f, 0);            // General purpose bit flag
        write_u16_le(f, 0);            // Compression method (STORED)
        write_u16_le(f, 0);            // File last modification time
        write_u16_le(f, 0);            // File last modification date
        write_u32_le(f, 0);            // CRC-32 (will update for correct impl)
        write_u32_le(f, data_size);    // Compressed size
        write_u32_le(f, data_size);    // Uncompressed size
        write_u16_le(f, static_cast<uint16_t>(name.size()));  // File name length
        write_u16_le(f, 0);            // Extra field length
        
        // File name
        f.write(name.c_str(), name.size());
        
        // File data
        f.write(data.data(), data.size());
        
        central_dir_entries.push_back({name, offset, data_size});
    }
    
    // Central directory
    uint32_t central_dir_start = static_cast<uint32_t>(f.tellp());
    
    for (const auto& [name, offset, size] : central_dir_entries) {
        write_u32_le(f, 0x02014b50);  // Central directory file header signature
        write_u16_le(f, 20);           // Version made by
        write_u16_le(f, 20);           // Version needed to extract
        write_u16_le(f, 0);            // General purpose bit flag
        write_u16_le(f, 0);            // Compression method
        write_u16_le(f, 0);            // File last modification time
        write_u16_le(f, 0);            // File last modification date
        write_u32_le(f, 0);            // CRC-32
        write_u32_le(f, size);         // Compressed size
        write_u32_le(f, size);         // Uncompressed size
        write_u16_le(f, static_cast<uint16_t>(name.size()));  // File name length
        write_u16_le(f, 0);            // Extra field length
        write_u16_le(f, 0);            // File comment length
        write_u16_le(f, 0);            // Disk number start
        write_u16_le(f, 0);            // Internal file attributes
        write_u32_le(f, 0);            // External file attributes
        write_u32_le(f, offset);       // Relative offset of local header
        
        // File name
        f.write(name.c_str(), name.size());
    }
    
    uint32_t central_dir_end = static_cast<uint32_t>(f.tellp());
    uint32_t central_dir_size = central_dir_end - central_dir_start;
    
    // End of central directory record
    write_u32_le(f, 0x06054b50);      // End of central directory signature
    write_u16_le(f, 0);                // Number of this disk
    write_u16_le(f, 0);                // Disk where central directory starts
    write_u16_le(f, static_cast<uint16_t>(central_dir_entries.size()));  // Number of central directory records on this disk
    write_u16_le(f, static_cast<uint16_t>(central_dir_entries.size()));  // Total number of central directory records
    write_u32_le(f, central_dir_size); // Size of central directory
    write_u32_le(f, central_dir_start); // Offset of start of central directory
    write_u16_le(f, 0);                // Comment length
}

} // namespace npy
