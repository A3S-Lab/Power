use candle_core::{Device, Tensor};

use crate::error::{PowerError, Result};

use super::super::plan::GraphNode;

pub(super) fn convolution_pads(
    node: &GraphNode,
    dimensions: (usize, usize, usize, usize),
    kernel: (usize, usize),
    stride: (usize, usize),
    dilation: (usize, usize),
) -> Result<(usize, usize, usize, usize)> {
    match node.string("auto_pad", "NOTSET")? {
        "NOTSET" => quad(&node.ints("pads", &[0, 0, 0, 0])?, "pads", node),
        "SAME_UPPER" => {
            let (_, _, height, width) = dimensions;
            let (top, bottom) = same_upper_padding(height, kernel.0, stride.0, dilation.0);
            let (left, right) = same_upper_padding(width, kernel.1, stride.1, dilation.1);
            Ok((top, left, bottom, right))
        }
        other => Err(execution_error(
            node,
            format!("unsupported auto_pad '{other}'"),
        )),
    }
}

pub(super) fn pool_pads(
    node: &GraphNode,
    dimensions: (usize, usize, usize, usize),
    kernel: (usize, usize),
    stride: (usize, usize),
) -> Result<(usize, usize, usize, usize)> {
    convolution_pads(node, dimensions, kernel, stride, (1, 1))
}

pub(super) fn same_upper_padding(
    input: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
) -> (usize, usize) {
    let output = input.div_ceil(stride);
    let effective = dilation * (kernel.saturating_sub(1)) + 1;
    let total = ((output.saturating_sub(1)) * stride + effective).saturating_sub(input);
    (total / 2, total - total / 2)
}

pub(super) fn pad_spatial(
    input: &Tensor,
    pads: (usize, usize, usize, usize),
    node: &GraphNode,
) -> Result<Tensor> {
    input
        .pad_with_zeros(2, pads.0, pads.2)
        .and_then(|value| value.pad_with_zeros(3, pads.1, pads.3))
        .map_err(|error| execution_error(node, error))
}

pub(super) fn subsample_spatial(
    input: &Tensor,
    stride: (usize, usize),
    device: &Device,
    node: &GraphNode,
) -> Result<Tensor> {
    let mut output = input.clone();
    for (axis, step) in [(2, stride.0), (3, stride.1)] {
        if step == 1 {
            continue;
        }
        let length = output
            .dim(axis)
            .map_err(|error| execution_error(node, error))?;
        let indices = (0..length)
            .step_by(step)
            .map(u32::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|_| execution_error(node, "subsample index exceeds u32"))?;
        let indices = Tensor::from_vec(indices.clone(), indices.len(), device)
            .map_err(|error| execution_error(node, error))?;
        output = output
            .index_select(&indices, axis)
            .map_err(|error| execution_error(node, error))?;
    }
    Ok(output)
}

pub(super) fn slice_tensor(
    input: &Tensor,
    axis: usize,
    start: i64,
    end: i64,
    step: i64,
    device: &Device,
    node: &GraphNode,
) -> Result<Tensor> {
    if step <= 0 {
        return Err(execution_error(node, "Slice step must be positive"));
    }
    let length = input
        .dim(axis)
        .map_err(|error| execution_error(node, error))?;
    let (start, end) = slice_bounds(length, start, end, node)?;
    if step == 1 {
        return input
            .narrow(axis, start, end - start)
            .map_err(|error| execution_error(node, error));
    }
    let step = usize::try_from(step).map_err(|_| execution_error(node, "invalid Slice step"))?;
    let indices = (start..end)
        .step_by(step)
        .map(u32::try_from)
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|_| execution_error(node, "Slice index exceeds u32"))?;
    let indices = Tensor::from_vec(indices.clone(), indices.len(), device)
        .map_err(|error| execution_error(node, error))?;
    input
        .index_select(&indices, axis)
        .map_err(|error| execution_error(node, error))
}

pub(super) fn slice_bounds(
    length: usize,
    start: i64,
    end: i64,
    node: &GraphNode,
) -> Result<(usize, usize)> {
    let length_i64 = i64::try_from(length).map_err(|_| execution_error(node, "axis too large"))?;
    let normalize = |value: i64| {
        if value < 0 {
            (length_i64 + value).max(0)
        } else {
            value.min(length_i64)
        }
    };
    let start = normalize(start);
    let end = normalize(end);
    if end < start {
        return Err(execution_error(node, "Slice end precedes start"));
    }
    Ok((start as usize, end as usize))
}

pub(super) fn resolve_reshape(
    input: &[usize],
    requested: &[i64],
    node: &GraphNode,
) -> Result<Vec<usize>> {
    if requested.is_empty() {
        return Err(execution_error(node, "Reshape target must not be empty"));
    }
    let input_elements = input.iter().product::<usize>();
    let mut output = Vec::with_capacity(requested.len());
    let mut inferred = None;
    let mut known = 1_usize;
    for (index, dimension) in requested.iter().copied().enumerate() {
        match dimension {
            -1 if inferred.is_none() => {
                inferred = Some(index);
                output.push(1);
            }
            0 if index < input.len() => {
                known = known
                    .checked_mul(input[index])
                    .ok_or_else(|| execution_error(node, "Reshape dimensions overflowed"))?;
                output.push(input[index]);
            }
            value if value > 0 => {
                let value = usize::try_from(value)
                    .map_err(|_| execution_error(node, "invalid Reshape dimension"))?;
                known = known
                    .checked_mul(value)
                    .ok_or_else(|| execution_error(node, "Reshape dimensions overflowed"))?;
                output.push(value);
            }
            _ => return Err(execution_error(node, "invalid Reshape target")),
        }
    }
    if let Some(index) = inferred {
        if known == 0 || !input_elements.is_multiple_of(known) {
            return Err(execution_error(node, "Reshape target cannot be inferred"));
        }
        output[index] = input_elements / known;
    } else if known != input_elements {
        return Err(execution_error(node, "Reshape changes the element count"));
    }
    Ok(output)
}

pub(super) fn normalized_axes(axes: &[i64], rank: usize, node: &GraphNode) -> Result<Vec<usize>> {
    axes.iter()
        .map(|axis| axis_index(*axis, rank, node))
        .collect()
}

pub(super) fn axis_index(axis: i64, rank: usize, node: &GraphNode) -> Result<usize> {
    let rank_i64 = i64::try_from(rank).map_err(|_| execution_error(node, "rank exceeds i64"))?;
    let axis = if axis < 0 { rank_i64 + axis } else { axis };
    if axis < 0 || axis >= rank_i64 {
        return Err(execution_error(
            node,
            format!("axis {axis} is out of range for rank {rank}"),
        ));
    }
    Ok(axis as usize)
}

pub(super) fn pair(values: &[i64], name: &str, node: &GraphNode) -> Result<(usize, usize)> {
    if values.len() != 2 {
        return Err(execution_error(
            node,
            format!("{name} must contain two values"),
        ));
    }
    Ok((
        positive_usize(values[0], name, node)?,
        positive_usize(values[1], name, node)?,
    ))
}

pub(super) fn quad(
    values: &[i64],
    name: &str,
    node: &GraphNode,
) -> Result<(usize, usize, usize, usize)> {
    if values.len() != 4 {
        return Err(execution_error(
            node,
            format!("{name} must contain four values"),
        ));
    }
    Ok((
        nonnegative_usize(values[0], name, node)?,
        nonnegative_usize(values[1], name, node)?,
        nonnegative_usize(values[2], name, node)?,
        nonnegative_usize(values[3], name, node)?,
    ))
}

pub(super) fn positive_usize(value: i64, name: &str, node: &GraphNode) -> Result<usize> {
    if value <= 0 {
        return Err(execution_error(node, format!("{name} must be positive")));
    }
    usize::try_from(value).map_err(|_| execution_error(node, format!("{name} exceeds usize")))
}

pub(super) fn nonnegative_usize(value: i64, name: &str, node: &GraphNode) -> Result<usize> {
    if value < 0 {
        return Err(execution_error(
            node,
            format!("{name} must be non-negative"),
        ));
    }
    usize::try_from(value).map_err(|_| execution_error(node, format!("{name} exceeds usize")))
}

pub(super) fn execution_error(node: &GraphNode, error: impl std::fmt::Display) -> PowerError {
    PowerError::InferenceFailed(format!(
        "static graph node '{}' ({:?}) failed: {error}",
        node.name, node.op
    ))
}
