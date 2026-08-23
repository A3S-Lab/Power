function Resolve-Qwen38QualityProfile {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [ValidateSet(
            'pure-q6',
            'prefix-fr-release',
            'full-vocabulary-current',
            'dspark-q4'
        )]
        [string]$Profile
    )

    if ($Profile -eq 'pure-q6') {
        return [pscustomobject][ordered]@{
            config_relative_path = 'quality\full-vocabulary-current.acl'
            requires_tbq4 = $false
            q6_external_draft_kind = $null
            modes = @(
                [pscustomobject][ordered]@{
                    label = 'q6-off'
                    model_role = 'q6'
                    spec_mode = 'off'
                    fr_vocab_size = $null
                    external_draft_kind = $null
                },
                [pscustomobject][ordered]@{
                    label = 'q6-mtp-full-vocab'
                    model_role = 'q6'
                    spec_mode = 'mtp'
                    fr_vocab_size = $null
                    external_draft_kind = $null
                }
            )
            comparisons = @(, @('q6-off', 'q6-mtp-full-vocab'))
        }
    }

    if ($Profile -eq 'dspark-q4') {
        return [pscustomobject][ordered]@{
            config_relative_path = 'dspark\quality-k10-s6.acl'
            requires_tbq4 = $false
            q6_external_draft_kind = 'dspark'
            modes = @(
                [pscustomobject][ordered]@{
                    label = 'q6-off'
                    model_role = 'q6'
                    spec_mode = 'off'
                    fr_vocab_size = $null
                    external_draft_kind = 'dspark'
                },
                [pscustomobject][ordered]@{
                    label = 'q6-dspark'
                    model_role = 'q6'
                    spec_mode = 'dspark'
                    fr_vocab_size = $null
                    external_draft_kind = 'dspark'
                }
            )
            comparisons = @(, @('q6-off', 'q6-dspark'))
        }
    }

    $mtpMode = if ($Profile -eq 'full-vocabulary-current') {
        [pscustomobject][ordered]@{
            label = 'tbq4-mtp-full-vocab'
            model_role = 'tbq4'
            spec_mode = 'mtp'
            fr_vocab_size = $null
            external_draft_kind = $null
        }
    } else {
        [pscustomobject][ordered]@{
            label = 'tbq4-mtp-fr'
            model_role = 'tbq4'
            spec_mode = 'mtp'
            fr_vocab_size = 8192
            external_draft_kind = $null
        }
    }
    $configRelativePath = if ($Profile -eq 'full-vocabulary-current') {
        'quality\full-vocabulary-current.acl'
    } else {
        'quality\matrix.acl'
    }

    [pscustomobject][ordered]@{
        config_relative_path = $configRelativePath
        requires_tbq4 = $true
        q6_external_draft_kind = $null
        modes = @(
            [pscustomobject][ordered]@{
                label = 'q6-off'
                model_role = 'q6'
                spec_mode = 'off'
                fr_vocab_size = $null
                external_draft_kind = $null
            },
            [pscustomobject][ordered]@{
                label = 'tbq4-off'
                model_role = 'tbq4'
                spec_mode = 'off'
                fr_vocab_size = $null
                external_draft_kind = $null
            },
            $mtpMode
        )
        comparisons = @(
            @('q6-off', 'tbq4-off'),
            @('tbq4-off', $mtpMode.label)
        )
    }
}
