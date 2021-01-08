#pragma once

#include <array>
#include <algorithm>

#include <cstdio>
#include <cstdint>

class PPM {
  bool      m_opened;

  uint64_t  m_width;       /**< @brief Image width in pixels.*/
  uint64_t  m_height;      /**< @brief Image height in pixels.*/
  uint16_t  m_max_value; /**< @brief Maximum RGB value of an image.*/

  void     *m_file;
  size_t    m_header_offset;

  int parseHeader(FILE *ppm);

  enum HeaderParserState {
    STATE_INIT,
    STATE_P,
    STATE_P6,
    STATE_P6_SPACE,
    STATE_WIDTH,
    STATE_WIDTH_SPACE,
    STATE_HEIGHT,
    STATE_HEIGHT_SPACE,
    STATE_MAX,
    STATE_END
  };

public:
  PPM()                       = default;
  PPM(const PPM &)            = delete;
  PPM &operator=(const PPM &) = delete;

  PPM(PPM &&);
  ~PPM();

  int createPPM(const char *file_name, uint64_t width, uint64_t height, uint16_t max_value);

  int mmapPPM(const char *file_name);

  void *data() const {
    return static_cast<uint8_t *>(m_file) + m_header_offset;
  }

  uint16_t get(size_t index) const {
    if (m_max_value > 255) {
      uint16_t value = static_cast<const uint16_t *>(data())[index];

      #if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
      uint8_t *ptr = reinterpret_cast<uint8_t *>(&value);
      std::reverse(ptr, ptr + sizeof(uint16_t));
      #endif

      return value;
    }
    else {
      return static_cast<const uint8_t *>(data())[index];
    }
  }

  void put(size_t index, uint16_t value) {
    if (m_max_value > 255) {
      #if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
      uint8_t *ptr = reinterpret_cast<uint8_t *>(&value);
      std::reverse(ptr, ptr + sizeof(uint16_t));
      #endif

      static_cast<uint16_t *>(data())[index] = value;
    }
    else {
      static_cast<uint8_t *>(data())[index] = value;
    }
  }

  uint64_t width() const {
    return m_width;
  }

  uint64_t height() const {
    return m_height;
  }

  uint32_t max_value() const {
    return m_max_value;
  }
};
