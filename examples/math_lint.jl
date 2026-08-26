# Lints the LaTeX in the markdown that Weave generates from a notebook.
#
# Why this exists: math in these examples is typeset by KaTeX in the reader's browser, not at
# build time, so a bad formula produces no build error at all - it just shows up as raw
# `$...$` / `\[...\]` text on the published page. Worse, KaTeX's `auto-render` only catches
# `ParseError`; any other exception propagates out of its DOM walk (see `renderElem` in
# `contrib/auto-render.js`), which leaves *every formula after the offending one on the page*
# unrendered. So a single unbalanced brace in one cell can blank out a whole tutorial.
#
# The rules below come from the "Mathematical Content" section of `docs/src/how_to_contribute.md`,
# plus the structural checks (delimiter termination, brace and environment balance) that catch the
# constructs KaTeX throws hard on.
#
# Note what this canNOT catch: math that is perfectly valid but never gets rendered because some
# unrelated script broke `documenter.js` before KaTeX ran. Only loading a built page in a real
# browser catches that - see `docs/tools/check-rendered-math.mjs`.

struct MathProblem
    line::Int
    severity::Symbol   # :error fails the build, :warning is a style nudge
    rule::String
    message::String
    snippet::String
end

struct MathSpan
    line::Int
    display::Bool
    content::String
    terminated::Bool
end

const MATH_SNIPPET_LIMIT = 120

function _truncate_snippet(s::AbstractString)
    oneline = replace(strip(s), r"\s*\n\s*" => " ⏎ ")
    chars = collect(oneline)
    return length(chars) <= MATH_SNIPPET_LIMIT ? oneline : String(chars[1:MATH_SNIPPET_LIMIT]) * " ..."
end

"""
    mask_code(content) -> String

Blank out fenced code blocks and inline code spans, keeping every newline in place so that line
numbers reported against the result still match the file on disk. Julia code is full of `\$` from
string interpolation (`"step \$k"`), and none of it is math.
"""
function mask_code(content::AbstractString)
    masked = IOBuffer()
    infence = false
    for line in split(content, '\n'; keepempty=true)
        stripped = lstrip(line)
        if startswith(stripped, "```") || startswith(stripped, "~~~")
            infence = !infence
            println(masked, "")
            continue
        end
        # Inline code spans cannot span lines, so it is safe to blank them per line. Keep the
        # backticks and the run length so column-ish context stays roughly honest.
        println(masked, infence ? "" : replace(line, r"`[^`]*`" => m -> "`" * repeat("_", max(0, length(collect(m)) - 2)) * "`"))
    end
    out = String(take!(masked))
    # `split`/`println` round-trip appends one trailing newline; drop it to preserve length.
    return endswith(out, '\n') && !endswith(content, '\n') ? chop(out) : out
end

"""
    find_math_spans(masked) -> Vector{MathSpan}

Locate every `\$...\$` and `\$\$...\$\$` span. `masked` must already have code blanked out.

An inline span is not allowed to run past a blank line: markdown cannot carry inline math across a
paragraph break, so hitting one means the opening `\$` was never closed. Bounding the scan this way
turns a stray dollar sign into a precise "unterminated" report instead of one giant bogus span.
"""
function find_math_spans(masked::AbstractString)
    chars = collect(masked)
    n = length(chars)

    linenos = Vector{Int}(undef, n)
    ln = 1
    for i in 1:n
        linenos[i] = ln
        chars[i] == '\n' && (ln += 1)
    end

    spans = MathSpan[]

    i = 1
    while i <= n
        c = chars[i]
        if c == '\\'
            i += 2                      # skip the escaped character, e.g. a literal \$
            continue
        elseif c != '$'
            i += 1
            continue
        end

        display = i < n && chars[i + 1] == '$'
        open_len = display ? 2 : 1
        j = i + open_len
        close_at = 0

        while j <= n
            cj = chars[j]
            if cj == '\\'
                j += 2
                continue
            elseif cj == '$'
                if display
                    if j < n && chars[j + 1] == '$'
                        close_at = j
                        break
                    end
                    j += 1             # a lone $ inside display math is not a terminator
                    continue
                end
                close_at = j
                break
            elseif !display && cj == '\n' && j < n && chars[j + 1] == '\n'
                break                  # paragraph break: inline math cannot reach past it
            end
            j += 1
        end

        if close_at == 0
            push!(spans, MathSpan(linenos[i], display, String(chars[i:min(n, i + 60)]), false))
            i += open_len
            continue
        end

        content = String(chars[(i + open_len):(close_at - 1)])

        push!(spans, MathSpan(linenos[i], display, content, true))
        i = close_at + open_len
    end

    return spans
end

# Count braces, ignoring escaped ones. `\{` is a literal brace, not a group delimiter.
function _brace_balance(content::AbstractString)
    chars = collect(content)
    depth = 0
    lowest = 0
    i = 1
    while i <= length(chars)
        c = chars[i]
        if c == '\\'
            i += 2
            continue
        elseif c == '{'
            depth += 1
        elseif c == '}'
            depth -= 1
            lowest = min(lowest, depth)
        end
        i += 1
    end
    return depth, lowest
end

function _environment_balance(content::AbstractString)
    opened = [m.captures[1] for m in eachmatch(r"\\begin\{([A-Za-z*]+)\}", content)]
    closed = [m.captures[1] for m in eachmatch(r"\\end\{([A-Za-z*]+)\}", content)]
    unclosed = String[]
    remaining = copy(closed)
    for env in reverse(opened)
        idx = findfirst(==(env), remaining)
        isnothing(idx) ? push!(unclosed, env) : deleteat!(remaining, idx)
    end
    return unclosed, remaining
end

"""
    lint_math(content) -> Vector{MathProblem}

Check the math in a generated markdown document. Returns one entry per problem found.
"""
function lint_math(content::AbstractString)
    problems = MathProblem[]
    for span in find_math_spans(mask_code(content))
        delim = span.display ? "\$\$" : "\$"
        snippet = _truncate_snippet(span.content)

        if !span.terminated
            push!(problems, MathProblem(span.line, :error, "unterminated-delimiter",
                "opening `$delim` is never closed. A stray dollar sign in prose will do this; " *
                "escape it as `\\\$` if it is not math.", snippet))
            continue
        end

        if isempty(strip(span.content))
            push!(problems, MathProblem(span.line, :error, "empty-math",
                "empty `$delim...$delim` span. Write a literal dollar sign as `\\\$`.", snippet))
            continue
        end

        # Prose caught between two unescaped dollar signs: "it costs $0.25 and $100 a month" is
        # read as one equation, and the dollars plus everything between them are lost.
        if !span.display && !occursin(r"[\\_^{}=+*/<>|]", span.content) && count(==(' '), span.content) >= 2
            push!(problems, MathProblem(span.line, :error, "prose-between-dollars",
                "this reads as an equation but looks like prose - two unescaped dollar signs in " *
                "a sentence pair up, and both signs and the text between them are dropped. " *
                "Escape a literal one as \\\$.", snippet))
            continue
        end

        # docs/src/how_to_contribute.md: "No space nor line breaks after opening $$ or $".
        # This one is load-bearing, not cosmetic. Documenter's markdown parser does not recognise
        # `$$\nx = 1\n$$` or `$ A $` as math at all, so the delimiters and the LaTeX are emitted
        # as ordinary prose and the reader sees the source. Verified against Documenter 1.17.
        if startswith(span.content, r"\s") || endswith(span.content, r"\s")
            push!(problems, MathProblem(span.line, :error, "delimiter-whitespace",
                "whitespace or a line break directly inside `$delim...$delim`. Documenter does not " *
                "read this as math - the delimiters and the LaTeX are published as plain text. " *
                "Put the LaTeX flush against both delimiters.", snippet))
        end

        if !span.display && occursin('\n', span.content)
            push!(problems, MathProblem(span.line, :warning, "inline-math-line-break",
                "inline `\$...\$` spans a line break. Keep inline math on one line, or promote it " *
                "to a display equation.", snippet))
        end

        # These are the ones that matter most: KaTeX raises a non-ParseError on them, and
        # auto-render then aborts its DOM walk, so *every* formula further down the page stays raw.
        depth, lowest = _brace_balance(span.content)
        if depth != 0 || lowest < 0
            push!(problems, MathProblem(span.line, :error, "unbalanced-braces",
                "unbalanced `{}` in math (net depth $depth). KaTeX throws a non-ParseError here, " *
                "which stops it rendering every later formula on the page too.", snippet))
        end

        unclosed, unopened = _environment_balance(span.content)
        if !isempty(unclosed) || !isempty(unopened)
            detail = String[]
            isempty(unclosed) || push!(detail, "never closed: " * join(unique(unclosed), ", "))
            isempty(unopened) || push!(detail, "closed but never opened: " * join(unique(unopened), ", "))
            push!(problems, MathProblem(span.line, :error, "unbalanced-environment",
                "mismatched LaTeX environment (" * join(detail, "; ") * "). Same failure mode as " *
                "unbalanced braces: it stops all later math on the page from rendering.", snippet))
        end
    end
    return problems
end

"""
    check_math(md_path) -> Bool

`true` when the generated markdown has math that will not render, mirroring `has_error_blocks` so
the caller can mark the example as failed. Style-level findings are logged as warnings and do not
fail the build.
"""
# `is_processing_failed` is evaluated more than once per notebook while the build reports its
# results, so remember the verdict - otherwise the same error block is printed twice. Only the
# driver process calls this, so a plain Dict is enough.
const MATH_CHECK_CACHE = Dict{String,Bool}()

check_math(md_path) = get!(() -> _check_math(md_path), MATH_CHECK_CACHE, string(md_path))

function _check_math(md_path)
    !isfile(md_path) && return false
    problems = lint_math(read(md_path, String))
    isempty(problems) && return false

    example = basename(dirname(md_path))
    format(ps) = join(["  $(md_path):$(p.line) [$(p.rule)]\n      $(p.message)\n      in: $(p.snippet)" for p in ps], "\n\n")

    warnings = filter(p -> p.severity === :warning, problems)
    if !isempty(warnings)
        @warn """
        Math style issues in $example (not fatal, but please fix):

        $(format(warnings))

        See the "Mathematical Content" rules in docs/src/how_to_contribute.md.
        """
    end

    errors = filter(p -> p.severity === :error, problems)
    isempty(errors) && return false

    @error """
    Broken LaTeX in $example.

    $(format(errors))

    This math reaches the published page as raw text instead of typeset formulas. KaTeX runs in the
    reader's browser, so nothing else in the build can catch it - and an unbalanced brace or
    environment stops every *later* formula on the page from rendering too.

    See the "Mathematical Content" rules in docs/src/how_to_contribute.md.
    """
    return true
end
