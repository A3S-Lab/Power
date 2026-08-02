use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;
use zeroize::{Zeroize, ZeroizeOnDrop, Zeroizing};

use crate::error::{PowerError, Result};

pub const LOSSLESS_RANS_FORMAT_METADATA_KEY: &str = "a3s.power.weight-representation";
pub const LOSSLESS_RANS_TABLE_METADATA_KEY: &str = "a3s.power.rans-nibble-256-v1.table";

const TABLE_SCHEMA: &str = "a3s.power.rans-nibble-256-table.v1";
const STREAMS: usize = 256;
const ALPHABET: usize = 16;
const SCALE_BITS: u32 = 14;
const SCALE: u32 = 1 << SCALE_BITS;
const RANS_L: u32 = 1 << 23;
const RANS_UPPER: u32 = RANS_L << 8;
const MAX_SCALE_BITS: u32 = 15;
const HEADER_UNPADDED_BYTES: usize = 16 + (STREAMS + 1) * 4;
const HEADER_BYTES: usize = round_16(HEADER_UNPADDED_BYTES);
const DECODE_CANCEL_INTERVAL: u64 = 4_096;

/// Zeroizing bytes for one encoded lossless record.
///
/// The wrapper deliberately exposes no serialization implementation and
/// redacts record contents from normal debug output.
pub struct LosslessEncodedRecord {
    bytes: Zeroizing<Vec<u8>>,
}

impl LosslessEncodedRecord {
    pub fn as_slice(&self) -> &[u8] {
        self.bytes.as_slice()
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

impl std::ops::Deref for LosslessEncodedRecord {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl std::fmt::Debug for LosslessEncodedRecord {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LosslessEncodedRecord")
            .field("bytes", &self.bytes.len())
            .finish_non_exhaustive()
    }
}

#[derive(Clone, PartialEq, Eq, Zeroize, ZeroizeOnDrop)]
pub struct LosslessRansNibbleTable {
    frequencies: [u32; ALPHABET],
    starts: [u32; ALPHABET],
    slot_to_symbol: Vec<u8>,
}

impl LosslessRansNibbleTable {
    pub const FORMAT: &'static str = "a3s.power.rans-nibble-256-v1";

    pub fn from_frequencies(frequencies: [u32; ALPHABET]) -> Result<Self> {
        let sum = frequencies.iter().try_fold(0_u32, |total, frequency| {
            total.checked_add(*frequency).ok_or_else(|| {
                PowerError::InvalidFormat(
                    "lossless rANS table frequency sum overflowed".to_string(),
                )
            })
        })?;
        if sum != SCALE {
            return Err(PowerError::InvalidFormat(format!(
                "lossless rANS table frequencies sum to {sum}; expected {SCALE}"
            )));
        }

        let mut starts = [0_u32; ALPHABET];
        let mut cursor = 0_u32;
        let mut slot_to_symbol = vec![0_u8; SCALE as usize];
        for (symbol, frequency) in frequencies.iter().copied().enumerate() {
            starts[symbol] = cursor;
            let end = cursor.checked_add(frequency).ok_or_else(|| {
                PowerError::InvalidFormat("lossless rANS table range overflowed".to_string())
            })?;
            let start_index = cursor as usize;
            let end_index = end as usize;
            slot_to_symbol[start_index..end_index].fill(symbol as u8);
            cursor = end;
        }
        Ok(Self {
            frequencies,
            starts,
            slot_to_symbol,
        })
    }

    pub fn frequencies(&self) -> &[u32; ALPHABET] {
        &self.frequencies
    }

    pub fn safetensors_metadata(&self) -> Result<HashMap<String, String>> {
        Ok(HashMap::from([
            (
                LOSSLESS_RANS_FORMAT_METADATA_KEY.to_string(),
                Self::FORMAT.to_string(),
            ),
            (
                LOSSLESS_RANS_TABLE_METADATA_KEY.to_string(),
                serde_json::to_string(&TableMetadata {
                    schema: TABLE_SCHEMA.to_string(),
                    streams: STREAMS as u32,
                    scale_bits: SCALE_BITS,
                    frequencies: self.frequencies,
                })?,
            ),
        ]))
    }

    pub fn from_metadata_json(value: &str) -> Result<Self> {
        let metadata: TableMetadata = serde_json::from_str(value).map_err(|error| {
            PowerError::InvalidFormat(format!(
                "failed to parse lossless rANS table metadata: {error}"
            ))
        })?;
        if metadata.schema != TABLE_SCHEMA
            || metadata.streams != STREAMS as u32
            || metadata.scale_bits != SCALE_BITS
        {
            return Err(PowerError::InvalidFormat(
                "lossless rANS table metadata has an unsupported identity".to_string(),
            ));
        }
        Self::from_frequencies(metadata.frequencies)
    }

    pub(super) fn from_safetensors_metadata(metadata: &HashMap<String, String>) -> Result<Self> {
        match metadata.get(LOSSLESS_RANS_FORMAT_METADATA_KEY) {
            Some(format) if format == Self::FORMAT => {}
            Some(_) => {
                return Err(PowerError::InvalidFormat(
                    "lossless weight source carries an unsupported representation stamp"
                        .to_string(),
                ));
            }
            None => {
                return Err(PowerError::InvalidFormat(
                    "lossless weight source is missing its mandatory representation stamp"
                        .to_string(),
                ));
            }
        }
        let table = metadata
            .get(LOSSLESS_RANS_TABLE_METADATA_KEY)
            .ok_or_else(|| {
                PowerError::InvalidFormat(
                    "lossless weight source is missing its shard-local rANS table".to_string(),
                )
            })?;
        Self::from_metadata_json(table)
    }

    pub fn encode_record(&self, bytes: &[u8], scratch_limit: u64) -> Result<LosslessEncodedRecord> {
        if bytes.is_empty() {
            return Err(PowerError::InvalidRequest(
                "lossless rANS input must not be empty".to_string(),
            ));
        }
        let symbol_count = u64::try_from(bytes.len())
            .ok()
            .and_then(|length| length.checked_mul(2))
            .ok_or_else(|| {
                PowerError::InvalidRequest(
                    "lossless rANS input symbol count overflowed".to_string(),
                )
            })?;
        let record_bound = record_bound(symbol_count)?;
        let stream_symbols = symbol_count / STREAMS as u64 + 1;
        let stream_bound = encoded_stream_bound(stream_symbols, SCALE_BITS)?;
        require_scratch(
            record_bound
                .checked_add(stream_bound)
                .ok_or_else(scratch_overflow)?,
            scratch_limit,
        )?;

        let output_len = usize::try_from(record_bound).map_err(|_| scratch_overflow())?;
        let mut output = Zeroizing::new(vec![0_u8; output_len]);
        output[..8].copy_from_slice(&symbol_count.to_le_bytes());
        output[8..16].copy_from_slice(&(bytes.len() as u64).to_le_bytes());

        let stream_capacity = usize::try_from(stream_bound).map_err(|_| scratch_overflow())?;
        let mut stream_scratch = Zeroizing::new(vec![0_u8; stream_capacity]);
        let mut payload_cursor = 0_usize;
        for stream in 0..STREAMS {
            let encoded = encode_stream(self, bytes, symbol_count, stream, &mut stream_scratch)?;
            let payload_start = HEADER_BYTES
                .checked_add(payload_cursor)
                .ok_or_else(scratch_overflow)?;
            let payload_end = payload_start
                .checked_add(encoded.len())
                .ok_or_else(scratch_overflow)?;
            if payload_end > output.len() {
                return Err(scratch_overflow());
            }
            output[payload_start..payload_end].copy_from_slice(encoded);
            write_u32(
                &mut output,
                16 + stream * 4,
                u32::try_from(payload_cursor).map_err(|_| scratch_overflow())?,
            )?;
            payload_cursor = payload_cursor
                .checked_add(encoded.len())
                .ok_or_else(scratch_overflow)?;
        }
        write_u32(
            &mut output,
            16 + STREAMS * 4,
            u32::try_from(payload_cursor).map_err(|_| scratch_overflow())?,
        )?;
        let total = HEADER_BYTES
            .checked_add(round_16_checked(payload_cursor)?)
            .ok_or_else(scratch_overflow)?;
        if total > output.len() {
            return Err(scratch_overflow());
        }
        output[total..].zeroize();
        output.truncate(total);
        Ok(LosslessEncodedRecord { bytes: output })
    }

    pub fn decode_record(
        &self,
        record: &[u8],
        expected_bytes: u64,
        scratch_limit: u64,
    ) -> Result<Zeroizing<Vec<u8>>> {
        self.decode_record_with_cancellation(
            record,
            expected_bytes,
            scratch_limit,
            &CancellationToken::new(),
        )
    }

    pub fn decode_record_with_cancellation(
        &self,
        record: &[u8],
        expected_bytes: u64,
        scratch_limit: u64,
        cancellation: &CancellationToken,
    ) -> Result<Zeroizing<Vec<u8>>> {
        check_cancelled(cancellation)?;
        let parsed = ParsedRecord::parse(record, expected_bytes, scratch_limit)?;
        let output_len = usize::try_from(expected_bytes).map_err(|_| scratch_overflow())?;
        let mut output = Zeroizing::new(vec![0_u8; output_len]);
        let mut decoded_symbols = 0_u64;

        for stream in 0..STREAMS {
            let start = parsed.offsets[stream] as usize;
            let end = parsed.offsets[stream + 1] as usize;
            let input = &parsed.payload[start..end];
            let mut cursor = 4_usize;
            let mut state = u32::from_be_bytes(input[..4].try_into().map_err(|_| {
                PowerError::InvalidFormat(
                    "lossless rANS stream is missing its initial state".to_string(),
                )
            })?);
            if !(RANS_L..RANS_UPPER).contains(&state) {
                return Err(PowerError::InvalidFormat(
                    "lossless rANS stream initial state is outside the encoder interval"
                        .to_string(),
                ));
            }
            let stream_symbols = stream_symbol_count(parsed.symbol_count, stream);
            for index in 0..stream_symbols {
                if decoded_symbols.is_multiple_of(DECODE_CANCEL_INTERVAL) {
                    check_cancelled(cancellation)?;
                }
                let slot = state & (SCALE - 1);
                let symbol = self.slot_to_symbol[slot as usize] as usize;
                let next = u64::from(self.frequencies[symbol])
                    .checked_mul(u64::from(state >> SCALE_BITS))
                    .and_then(|value| value.checked_add(u64::from(slot)))
                    .and_then(|value| value.checked_sub(u64::from(self.starts[symbol])))
                    .ok_or_else(|| {
                        PowerError::InvalidFormat(
                            "lossless rANS stream state arithmetic overflowed".to_string(),
                        )
                    })?;
                state = u32::try_from(next).map_err(|_| {
                    PowerError::InvalidFormat(
                        "lossless rANS stream state exceeds 32 bits".to_string(),
                    )
                })?;
                while state < RANS_L && cursor < input.len() {
                    state = (state << 8) | u32::from(input[cursor]);
                    cursor += 1;
                }
                let logical = u64::try_from(stream)
                    .ok()
                    .and_then(|base| {
                        index
                            .checked_mul(STREAMS as u64)
                            .and_then(|value| value.checked_add(base))
                    })
                    .ok_or_else(|| {
                        PowerError::InvalidFormat(
                            "lossless rANS output position overflowed".to_string(),
                        )
                    })?;
                set_nibble(&mut output, logical, symbol as u8)?;
                decoded_symbols = decoded_symbols.checked_add(1).ok_or_else(|| {
                    PowerError::InvalidFormat(
                        "lossless rANS decoded symbol count overflowed".to_string(),
                    )
                })?;
            }
            if cursor != input.len() {
                return Err(PowerError::InvalidFormat(
                    "lossless rANS stream has unconsumed payload bytes".to_string(),
                ));
            }
            if state != RANS_L {
                return Err(PowerError::InvalidFormat(
                    "lossless rANS stream final state does not match the encoder origin"
                        .to_string(),
                ));
            }
        }
        if decoded_symbols != parsed.symbol_count {
            return Err(PowerError::InvalidFormat(
                "lossless rANS decoded symbol count is inconsistent".to_string(),
            ));
        }
        check_cancelled(cancellation)?;
        Ok(output)
    }
}

impl std::fmt::Debug for LosslessRansNibbleTable {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LosslessRansNibbleTable")
            .field("format", &Self::FORMAT)
            .field(
                "active_symbols",
                &self
                    .frequencies
                    .iter()
                    .filter(|frequency| **frequency != 0)
                    .count(),
            )
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Default, PartialEq, Eq, Zeroize, ZeroizeOnDrop)]
pub struct LosslessRansNibbleHistogram {
    counts: [u64; ALPHABET],
    bytes: u64,
}

impl LosslessRansNibbleHistogram {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut histogram = Self::default();
        histogram.observe(bytes)?;
        Ok(histogram)
    }

    pub fn observe(&mut self, bytes: &[u8]) -> Result<()> {
        self.bytes = self
            .bytes
            .checked_add(u64::try_from(bytes.len()).map_err(|_| histogram_overflow())?)
            .ok_or_else(histogram_overflow)?;
        for byte in bytes {
            for symbol in [byte & 0x0f, byte >> 4] {
                let count = &mut self.counts[symbol as usize];
                *count = count.checked_add(1).ok_or_else(histogram_overflow)?;
            }
        }
        Ok(())
    }

    pub fn build(&self) -> Result<LosslessRansNibbleTable> {
        if self.bytes == 0 {
            return Err(PowerError::InvalidRequest(
                "lossless rANS histogram is empty".to_string(),
            ));
        }
        LosslessRansNibbleTable::from_frequencies(normalize_frequencies(&self.counts)?)
    }
}

impl std::fmt::Debug for LosslessRansNibbleHistogram {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LosslessRansNibbleHistogram")
            .field("bytes", &self.bytes)
            .field(
                "active_symbols",
                &self.counts.iter().filter(|count| **count != 0).count(),
            )
            .finish_non_exhaustive()
    }
}

#[derive(Serialize, Deserialize, Zeroize, ZeroizeOnDrop)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct TableMetadata {
    schema: String,
    streams: u32,
    scale_bits: u32,
    frequencies: [u32; ALPHABET],
}

struct ParsedRecord<'a> {
    symbol_count: u64,
    offsets: [u32; STREAMS + 1],
    payload: &'a [u8],
}

impl<'a> ParsedRecord<'a> {
    fn parse(record: &'a [u8], expected_bytes: u64, scratch_limit: u64) -> Result<Self> {
        if expected_bytes == 0 || record.len() < HEADER_BYTES {
            return Err(PowerError::InvalidFormat(
                "lossless rANS record is empty or truncated".to_string(),
            ));
        }
        let record_bytes = u64::try_from(record.len()).map_err(|_| scratch_overflow())?;
        require_scratch(
            record_bytes
                .checked_add(expected_bytes)
                .ok_or_else(scratch_overflow)?,
            scratch_limit,
        )?;
        let symbol_count = read_u64(record, 0)?;
        let packed_bytes = read_u64(record, 8)?;
        let expected_symbols = expected_bytes.checked_mul(2).ok_or_else(scratch_overflow)?;
        if symbol_count != expected_symbols || packed_bytes != expected_bytes {
            return Err(PowerError::InvalidFormat(
                "lossless rANS record decoded size does not match the canonical tensor".to_string(),
            ));
        }

        let mut offsets = [0_u32; STREAMS + 1];
        for (index, offset) in offsets.iter_mut().enumerate() {
            *offset = read_u32(record, 16 + index * 4)?;
        }
        if offsets[0] != 0 {
            return Err(PowerError::InvalidFormat(
                "lossless rANS first stream offset is not zero".to_string(),
            ));
        }
        for pair in offsets.windows(2) {
            if pair[1] < pair[0] || pair[1] - pair[0] < 4 {
                return Err(PowerError::InvalidFormat(
                    "lossless rANS stream offsets are invalid".to_string(),
                ));
            }
        }
        let payload_bytes = offsets[STREAMS] as usize;
        let expected_record_bytes = HEADER_BYTES
            .checked_add(round_16_checked(payload_bytes)?)
            .ok_or_else(scratch_overflow)?;
        if expected_record_bytes != record.len() {
            return Err(PowerError::InvalidFormat(
                "lossless rANS record length does not match its framing".to_string(),
            ));
        }
        if record[HEADER_UNPADDED_BYTES..HEADER_BYTES]
            .iter()
            .any(|byte| *byte != 0)
            || record[HEADER_BYTES + payload_bytes..]
                .iter()
                .any(|byte| *byte != 0)
        {
            return Err(PowerError::InvalidFormat(
                "lossless rANS record padding is not zero".to_string(),
            ));
        }
        let amplification_bound = u64::try_from(payload_bytes)
            .ok()
            .and_then(|bytes| bytes.checked_mul(8))
            .and_then(|bits| bits.checked_mul(1_u64 << MAX_SCALE_BITS))
            .ok_or_else(scratch_overflow)?;
        if symbol_count > amplification_bound {
            return Err(PowerError::InvalidFormat(
                "lossless rANS record exceeds the codec amplification bound".to_string(),
            ));
        }
        Ok(Self {
            symbol_count,
            offsets,
            payload: &record[HEADER_BYTES..HEADER_BYTES + payload_bytes],
        })
    }
}

fn normalize_frequencies(counts: &[u64; ALPHABET]) -> Result<[u32; ALPHABET]> {
    let total = counts.iter().try_fold(0_u128, |sum, count| {
        sum.checked_add(u128::from(*count))
            .ok_or_else(histogram_overflow)
    })?;
    if total == 0 {
        return Err(PowerError::InvalidRequest(
            "lossless rANS histogram is empty".to_string(),
        ));
    }
    let mut frequencies = [0_u32; ALPHABET];
    let mut remainders = [0_u128; ALPHABET];
    for symbol in 0..ALPHABET {
        if counts[symbol] == 0 {
            continue;
        }
        let scaled = u128::from(counts[symbol])
            .checked_mul(u128::from(SCALE))
            .ok_or_else(histogram_overflow)?;
        frequencies[symbol] =
            u32::try_from((scaled / total).max(1)).map_err(|_| histogram_overflow())?;
        remainders[symbol] = scaled % total;
    }

    let mut sum = frequencies
        .iter()
        .map(|value| u64::from(*value))
        .sum::<u64>();
    while sum > u64::from(SCALE) {
        let symbol = (0..ALPHABET)
            .filter(|index| frequencies[*index] > 1)
            .max_by_key(|index| (frequencies[*index], std::cmp::Reverse(*index)))
            .ok_or_else(histogram_overflow)?;
        frequencies[symbol] -= 1;
        sum -= 1;
    }
    while sum < u64::from(SCALE) {
        let symbol = (0..ALPHABET)
            .filter(|index| counts[*index] != 0)
            .max_by_key(|index| (remainders[*index], std::cmp::Reverse(*index)))
            .ok_or_else(histogram_overflow)?;
        frequencies[symbol] = frequencies[symbol]
            .checked_add(1)
            .ok_or_else(histogram_overflow)?;
        remainders[symbol] = 0;
        sum += 1;
    }
    Ok(frequencies)
}

fn encode_stream<'a>(
    table: &LosslessRansNibbleTable,
    bytes: &[u8],
    symbol_count: u64,
    stream: usize,
    scratch: &'a mut [u8],
) -> Result<&'a [u8]> {
    scratch.zeroize();
    let mut cursor = scratch.len();
    let symbols = stream_symbol_count(symbol_count, stream);
    let mut state = RANS_L;
    for index in (0..symbols).rev() {
        let logical = index
            .checked_mul(STREAMS as u64)
            .and_then(|value| value.checked_add(stream as u64))
            .ok_or_else(scratch_overflow)?;
        let symbol = nibble_at(bytes, logical)? as usize;
        let frequency = table.frequencies[symbol];
        if frequency == 0 {
            return Err(PowerError::InvalidRequest(
                "lossless rANS table cannot encode an observed symbol".to_string(),
            ));
        }
        let maximum = ((RANS_L >> SCALE_BITS) << 8)
            .checked_mul(frequency)
            .ok_or_else(scratch_overflow)?;
        while state >= maximum {
            cursor = cursor.checked_sub(1).ok_or_else(scratch_overflow)?;
            scratch[cursor] = state as u8;
            state >>= 8;
        }
        let updated = ((state / frequency) << SCALE_BITS)
            .checked_add(state % frequency)
            .and_then(|value| value.checked_add(table.starts[symbol]))
            .ok_or_else(scratch_overflow)?;
        state = updated;
    }
    for _ in 0..4 {
        cursor = cursor.checked_sub(1).ok_or_else(scratch_overflow)?;
        scratch[cursor] = state as u8;
        state >>= 8;
    }
    Ok(&scratch[cursor..])
}

fn stream_symbol_count(symbol_count: u64, stream: usize) -> u64 {
    let stream = stream as u64;
    if symbol_count <= stream {
        0
    } else {
        (symbol_count - 1 - stream) / STREAMS as u64 + 1
    }
}

fn nibble_at(bytes: &[u8], logical: u64) -> Result<u8> {
    let byte_index = usize::try_from(logical / 2).map_err(|_| scratch_overflow())?;
    let byte = *bytes.get(byte_index).ok_or_else(|| {
        PowerError::InvalidFormat("lossless rANS input position is out of bounds".to_string())
    })?;
    Ok(if logical & 1 == 0 {
        byte & 0x0f
    } else {
        byte >> 4
    })
}

fn set_nibble(output: &mut [u8], logical: u64, symbol: u8) -> Result<()> {
    let byte_index = usize::try_from(logical / 2).map_err(|_| scratch_overflow())?;
    let byte = output.get_mut(byte_index).ok_or_else(|| {
        PowerError::InvalidFormat("lossless rANS output position is out of bounds".to_string())
    })?;
    if logical & 1 == 0 {
        *byte = (*byte & 0xf0) | symbol;
    } else {
        *byte = (*byte & 0x0f) | (symbol << 4);
    }
    Ok(())
}

fn record_bound(symbol_count: u64) -> Result<u64> {
    let per_stream = symbol_count / STREAMS as u64 + 1;
    let stream_bound = encoded_stream_bound(per_stream, SCALE_BITS)?;
    let payload = stream_bound
        .checked_mul(STREAMS as u64)
        .ok_or_else(scratch_overflow)?;
    u64::try_from(HEADER_BYTES)
        .ok()
        .and_then(|header| {
            round_16_u64_checked(payload).and_then(|payload| header.checked_add(payload))
        })
        .ok_or_else(scratch_overflow)
}

fn encoded_stream_bound(symbol_count: u64, scale_bits: u32) -> Result<u64> {
    symbol_count
        .checked_mul(u64::from(scale_bits))
        .and_then(|bits| bits.checked_add(7))
        .map(|bits| bits / 8)
        .and_then(|bytes| bytes.checked_add(4))
        .ok_or_else(scratch_overflow)
}

const fn round_16(value: usize) -> usize {
    (value + 15) & !15
}

fn round_16_checked(value: usize) -> Result<usize> {
    value
        .checked_add(15)
        .map(|adjusted| adjusted & !15)
        .ok_or_else(scratch_overflow)
}

fn round_16_u64_checked(value: u64) -> Option<u64> {
    value.checked_add(15).map(|adjusted| adjusted & !15)
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64> {
    let end = offset.checked_add(8).ok_or_else(scratch_overflow)?;
    bytes
        .get(offset..end)
        .and_then(|value| value.try_into().ok())
        .map(u64::from_le_bytes)
        .ok_or_else(|| PowerError::InvalidFormat("lossless rANS record is truncated".to_string()))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    let end = offset.checked_add(4).ok_or_else(scratch_overflow)?;
    bytes
        .get(offset..end)
        .and_then(|value| value.try_into().ok())
        .map(u32::from_le_bytes)
        .ok_or_else(|| PowerError::InvalidFormat("lossless rANS record is truncated".to_string()))
}

fn write_u32(bytes: &mut [u8], offset: usize, value: u32) -> Result<()> {
    let end = offset.checked_add(4).ok_or_else(scratch_overflow)?;
    let destination = bytes.get_mut(offset..end).ok_or_else(scratch_overflow)?;
    destination.copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn require_scratch(required: u64, limit: u64) -> Result<()> {
    if required > limit {
        Err(PowerError::InvalidRequest(format!(
            "lossless rANS scratch requires {required} bytes, exceeding the {limit} byte limit"
        )))
    } else {
        Ok(())
    }
}

fn check_cancelled(cancellation: &CancellationToken) -> Result<()> {
    if cancellation.is_cancelled() {
        Err(PowerError::InferenceFailed(
            "lossless weight decode was cancelled".to_string(),
        ))
    } else {
        Ok(())
    }
}

fn scratch_overflow() -> PowerError {
    PowerError::InvalidFormat("lossless rANS scratch arithmetic overflowed".to_string())
}

fn histogram_overflow() -> PowerError {
    PowerError::InvalidFormat("lossless rANS histogram arithmetic overflowed".to_string())
}
