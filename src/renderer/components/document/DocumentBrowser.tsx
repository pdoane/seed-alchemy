import { Box, Button, Flex, Text } from "@radix-ui/themes";
import { FileImageIcon, PlusIcon } from "@phosphor-icons/react";
import type { DocumentSummary } from "../../../shared/types/document";

interface DocumentBrowserProps {
  documents: DocumentSummary[];
  selectedDocumentId: string | null;
  onSelectDocument: (documentId: string) => void;
  onCreateDocument?: () => void;
}

export function DocumentBrowser({
  documents,
  selectedDocumentId,
  onSelectDocument,
  onCreateDocument,
}: DocumentBrowserProps) {
  return (
    <Flex direction="column" className="h-full bg-[var(--gray-2)]">
      <Flex
        align="center"
        justify="between"
        px="3"
        py="2"
        className="border-b border-[var(--gray-6)]"
      >
        <Text size="2" weight="medium">
          Documents
        </Text>
        {onCreateDocument && (
          <Button
            size="1"
            variant="ghost"
            color="gray"
            onClick={onCreateDocument}
          >
            <PlusIcon size={14} weight="bold" />
          </Button>
        )}
      </Flex>

      <Flex className="flex-1 gap-2 overflow-x-auto p-2">
        {documents.map((doc) => (
          <Flex
            key={doc.id}
            direction="column"
            align="center"
            gap="1"
            p="2"
            className={`cursor-pointer rounded transition-colors ${
              doc.id === selectedDocumentId
                ? "bg-[var(--violet-4)]"
                : "hover:bg-[var(--gray-4)]"
            }`}
            onClick={() => onSelectDocument(doc.id)}
          >
            <Box className="flex h-16 w-16 items-center justify-center rounded bg-[var(--gray-5)]">
              <FileImageIcon size={32} weight="fill" />
            </Box>
            <Text size="1" className="max-w-[80px] truncate">
              {doc.name ?? "Untitled"}
            </Text>
          </Flex>
        ))}

        {documents.length === 0 && (
          <Flex
            align="center"
            justify="center"
            className="w-full text-[var(--gray-9)]"
          >
            <Text size="1">No documents. Click + to create one.</Text>
          </Flex>
        )}
      </Flex>
    </Flex>
  );
}
