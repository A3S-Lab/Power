//! Typed parsing for the local Intel TDX `TDREPORT_STRUCT` layout.
//!
//! A TDREPORT is local, MAC-protected evidence. It is not a remotely
//! verifiable DCAP Quote, but fields exposed by the collector must still be
//! parsed from their architectural offsets.

pub(crate) const TDREPORT_BYTES: usize = 1024;

const REPORT_DATA_OFFSET: usize = 128;
const REPORT_DATA_BYTES: usize = 64;
const MRTD_OFFSET: usize = 528;
const MRTD_BYTES: usize = 48;

#[derive(Debug, Clone, Copy)]
pub(crate) struct TdxReportFields<'a> {
    pub(crate) report_data: &'a [u8],
    pub(crate) mrtd: &'a [u8],
}

pub(crate) fn parse_tdreport(raw: &[u8]) -> Option<TdxReportFields<'_>> {
    if raw.len() != TDREPORT_BYTES {
        return None;
    }

    Some(TdxReportFields {
        report_data: raw.get(REPORT_DATA_OFFSET..REPORT_DATA_OFFSET + REPORT_DATA_BYTES)?,
        mrtd: raw.get(MRTD_OFFSET..MRTD_OFFSET + MRTD_BYTES)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_report_data_and_mrtd_at_architectural_offsets() {
        let mut raw = [0_u8; TDREPORT_BYTES];
        raw[64..128].fill(0xee); // Poison the previously used wrong offset.
        raw[REPORT_DATA_OFFSET..REPORT_DATA_OFFSET + REPORT_DATA_BYTES].fill(0x11);
        raw[MRTD_OFFSET..MRTD_OFFSET + MRTD_BYTES].fill(0x22);

        let fields = parse_tdreport(&raw).expect("valid TDREPORT layout");
        assert_eq!(fields.report_data, &[0x11; REPORT_DATA_BYTES]);
        assert_eq!(fields.mrtd, &[0x22; MRTD_BYTES]);
    }

    #[test]
    fn rejects_non_tdreport_lengths() {
        assert!(parse_tdreport(&[0_u8; TDREPORT_BYTES - 1]).is_none());
        assert!(parse_tdreport(&[0_u8; TDREPORT_BYTES + 1]).is_none());
    }
}
