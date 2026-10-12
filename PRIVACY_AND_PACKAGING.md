# Privacy and packaging verification

The upload copy was prepared on 2026-10-12. The original study folder is preserved locally.

- All selected text files were checked for user-specific local paths and detected account/contact markers.
- Markdown links to retained local files were converted to relative repository paths; local helper references were removed from the navigation.
- PDF descriptive metadata, XMP metadata, embedded attachments and annotation-author metadata were removed. Local file links in PDFs were removed.
- PNG/JPEG metadata were stripped. PNG pixels were preserved.
- Complete official data ZIPs were retained and checked for CRC integrity.
- Virtual environments, caches, private audit markers, obsolete reports and local operating notes were not uploaded.
- Public third-party author/institution names, award identifiers, athlete names and public numerical records are source data, not the repository owner's personal information, and were retained.
- No previous local Git history is imported. The initial commit uses a neutral, non-personal identity.

Checksums describe the sanitized copies. Source hashes retained in the archive manifest refer to the original downloaded files. This process checks detected markers and metadata; it is not a guarantee that every unknown identifying string can be recognized. GitHub ownership remains linked to the repository account.
